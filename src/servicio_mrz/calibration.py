"""Autoevaluacion de la camara: busca los ajustes que dan la mejor foto para leer la MRZ.

Se usa solo cuando cambia el hardware (tecnico, con un documento de prueba quieto frente a la
camara). Principios:
- No se confia en que el driver "acepte" un valor: se mide si brillo/nitidez cambian de verdad.
- Los barridos usan metricas baratas (nitidez, brillo, reflejo); la decision final la toma la
  lectura real de la MRZ (OCR + digitos de control).
- Nunca empeora: si la configuracion calibrada no lee mejor que la inicial, se conserva la inicial.
- Nada de lo que registra contiene datos del documento.
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

TARGET_BRIGHTNESS = 135.0
MAX_GLARE_PCT = 1.5
EXPOSURE_SPAN = 4  # pasos a cada lado del valor inicial
FOCUS_GRID = tuple(range(0, 256, 16))
RESOLUTIONS = ((1280, 720), (1920, 1080))
READS_PER_CONFIG = 3
MIN_EFFECT_BRIGHTNESS = 4.0  # un ajuste de exposicion que no mueve el brillo no sirve
MIN_EFFECT_SHARPNESS = 0.05  # 5 %
METRIC_WIDTH = 640  # ancho al que se normaliza la zona central antes de medir nitidez
MIN_FPS = 10.0  # con menos cuadros/s, un documento que se mueve sale borroso
MIN_SHARPNESS_GAIN = 1.10  # para cambiar la referencia, la nitidez debe subir al menos 10 %


class CameraLike(Protocol):
    index: int
    width: int
    height: int

    def grab_settled(self, count: int = ...) -> list[np.ndarray]: ...
    def fps(self) -> float: ...
    def get(self, name: str) -> float | None: ...
    def apply(self, settings: dict[str, float]) -> dict[str, bool]: ...
    def switch(self, index: int, width: int | None = ..., height: int | None = ...) -> bool: ...
    def resolution(self) -> tuple[int, int] | None: ...


class ReaderLike(Protocol):
    def read(self, img: np.ndarray) -> Any: ...


def frame_metrics(frame: np.ndarray) -> dict[str, float]:
    """Nitidez (varianza del Laplaciano), brillo y % de pixeles quemados, en la zona central."""
    h, w = frame.shape[:2]
    center = frame[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)]
    scale = METRIC_WIDTH / w  # escala segun el ancho del CUADRO: 720p y 1080p quedan comparables
    if scale < 1.0:
        center = cv2.resize(center, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    return {
        "sharp": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "bright": float(gray.mean()),
        "glare": float((gray >= 250).mean() * 100),
    }


def _median_metrics(frames: list[np.ndarray]) -> dict[str, float] | None:
    if not frames:
        return None
    all_m = [frame_metrics(f) for f in frames]
    return {k: statistics.median(m[k] for m in all_m) for k in all_m[0]}


@dataclass
class State:
    status: str = "idle"  # idle | running | done | error | cancelled
    phase: str = ""
    progress: int = 0
    log: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None


class Calibrator:
    def __init__(self, camera: CameraLike, reader: ReaderLike, log_dir: Path | None = None) -> None:
        self.camera, self.reader = camera, reader
        self._log_dir = log_dir
        self.state = State()
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    # --- control -------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.state.status == "running"

    def start(self, on_done: Callable[[dict[str, Any]], None] | None = None) -> bool:
        with self._lock:
            if self.running:
                return False
            self.state = State(status="running", phase="inicio")
            self._cancel.clear()
        threading.Thread(target=self._run, args=(on_done,), name="calibration", daemon=True).start()
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def _say(self, msg: str, progress: int | None = None, phase: str | None = None) -> None:
        self.state.log.append(msg)
        if progress is not None:
            self.state.progress = progress
        if phase:
            self.state.phase = phase
        log.info("calibracion: %s", msg)

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise _Cancelled

    # --- capacidades ---------------------------------------------------------------------
    def capabilities(self) -> dict[str, dict[str, Any]]:
        """Que parametros controla de verdad la camara activa (mide el efecto, no la lectura)."""
        caps: dict[str, dict[str, Any]] = {}
        for name in ("exposure", "focus", "brightness", "contrast"):
            current = self.camera.get(name)
            # `efectivo` None = no evaluado (brillo y contraste no se barren: se ajustan a mano)
            measured = name in ("exposure", "focus")
            entry: dict[str, Any] = {
                "actual": current,
                "acepta": False,
                "efectivo": False if measured else None,
            }
            if name == "focus":
                # En DirectShow un enfoque que "no se puede leer" (-1) puede aun aceptar cambios:
                # no se descarta por la lectura, se mide el efecto real sobre la nitidez.
                entry["efectivo"] = self._focus_is_effective()
                entry["acepta"] = bool(entry["efectivo"])
            elif current is not None and current != -1.0:
                self.camera.apply({name: current + 1.0})
                entry["acepta"] = abs((self.camera.get(name) or current) - current) > 1e-6
                self.camera.apply({name: current})
                if name == "exposure" and entry["acepta"]:
                    entry["efectivo"] = self._exposure_is_effective(float(current))
            caps[name] = entry
        return caps

    def _exposure_is_effective(self, base: float) -> bool:
        dark = self._measure({"exposure": base - EXPOSURE_SPAN})
        bright = self._measure({"exposure": base + EXPOSURE_SPAN})
        self.camera.apply({"exposure": base})
        return bool(dark and bright and bright["bright"] - dark["bright"] >= MIN_EFFECT_BRIGHTNESS)

    def _focus_is_effective(self) -> bool:
        current = self.camera.get("focus")
        autofocus = self.camera.get("autofocus")
        self.camera.apply({"autofocus": 0, "focus": FOCUS_GRID[0]})
        near = self._measure({"focus": FOCUS_GRID[0]})
        far = self._measure({"focus": FOCUS_GRID[-1]})
        if current is not None and current != -1.0:
            self.camera.apply({"focus": current})
        # restaura el autoenfoque; si el driver no deja leerlo, el valor de fabrica es "activo"
        self.camera.apply({"autofocus": autofocus if autofocus not in (None, -1.0) else 1})
        if not near or not far or max(near["sharp"], far["sharp"]) == 0:
            return False
        change = abs(near["sharp"] - far["sharp"]) / max(near["sharp"], far["sharp"])
        return change >= MIN_EFFECT_SHARPNESS

    # --- medicion ------------------------------------------------------------------------
    def _measure(self, settings: dict[str, float] | None = None) -> dict[str, float] | None:
        if settings:
            self.camera.apply(settings)
        metrics = _median_metrics(self.camera.grab_settled(READS_PER_CONFIG))
        if metrics is not None:
            metrics["fps"] = self.camera.fps()
        return metrics

    def _evaluate(self, label: str) -> dict[str, Any]:
        """Lectura real de la MRZ (OCR) con la configuracion actual."""
        frames = self.camera.grab_settled(READS_PER_CONFIG)
        ok, times = 0, []
        for frame in frames:
            started = time.monotonic()
            result = self.reader.read(frame)
            times.append(time.monotonic() - started)
            ok += bool(result is not None and result.core_checks_ok)
        metrics = _median_metrics(frames) or {"sharp": 0.0, "bright": 0.0, "glare": 0.0}
        evaluation = {
            "etiqueta": label,
            "lecturas_ok": ok,
            "lecturas": len(frames),
            "ms_medio": int(statistics.mean(times) * 1000) if times else 0,
            **{k: round(v, 1) for k, v in metrics.items()},
            "fps": round(self.camera.fps(), 1),
        }
        self._say(f"{label}: {ok}/{len(frames)} lecturas, nitidez {evaluation['sharp']}")
        return evaluation

    # --- barridos ------------------------------------------------------------------------
    def _sweep_exposure(self, base: float) -> float:
        best: tuple[float, float] | None = None  # (puntaje, valor)
        for step in range(-EXPOSURE_SPAN, EXPOSURE_SPAN + 1):
            self._check_cancel()
            value = base + step
            m = self._measure({"exposure": value})
            if m is None:
                continue
            penalty = 1000.0 if m["glare"] > MAX_GLARE_PCT else 0.0
            penalty += 500.0 if m["fps"] < MIN_FPS else 0.0
            score = abs(m["bright"] - TARGET_BRIGHTNESS) + penalty - 0.001 * m["sharp"]
            if best is None or score < best[0]:
                best = (score, value)
        chosen = best[1] if best else base
        self.camera.apply({"exposure": chosen})
        self._say(f"exposicion elegida: {chosen:+.0f} (inicial {base:+.0f})")
        return chosen

    def _sweep_focus(self) -> float | None:
        self.camera.apply({"autofocus": 0})
        best: tuple[float, float] | None = None
        for value in FOCUS_GRID:
            self._check_cancel()
            m = self._measure({"focus": value})
            if m and (best is None or m["sharp"] > best[0]):
                best = (m["sharp"], value)
        if best is None:
            return None
        self.camera.apply({"focus": best[1]})
        self._say(f"enfoque elegido: {best[1]:.0f} (nitidez {best[0]:.0f})")
        return best[1]

    # --- proceso -------------------------------------------------------------------------
    def _run(self, on_done: Callable[[dict[str, Any]], None] | None) -> None:
        start_res = (self.camera.width, self.camera.height)
        start_settings: dict[str, float] = {}
        try:
            self._say("Midiendo capacidades de la camara...", 5, "capacidades")
            caps = self.capabilities()
            for name, entry in caps.items():
                self._say(f"{name}: acepta={entry['acepta']}, efectivo={entry['efectivo']}")
            start_settings = {}
            autofocus = self.camera.get("autofocus")
            if autofocus is not None and autofocus != -1.0:
                start_settings["autofocus"] = autofocus  # primero: el enfoque manual lo apaga
            start_settings.update(
                {n: e["actual"] for n, e in caps.items() if e["acepta"] and e["actual"] is not None}
            )

            self._say("Lectura de referencia (ajustes actuales)...", 15, "referencia")
            baseline = self._evaluate("referencia")

            chosen = dict(start_settings)
            if caps["exposure"]["efectivo"]:
                self._say("Barrido de exposicion...", 30, "exposicion")
                chosen["exposure"] = self._sweep_exposure(float(caps["exposure"]["actual"]))
            if caps["focus"]["efectivo"]:
                self._say("Barrido de enfoque...", 50, "enfoque")
                focus = self._sweep_focus()
                if focus is not None:
                    chosen["autofocus"] = 0
                    chosen["focus"] = focus

            self._say("Lectura con la configuracion calibrada...", 70, "validacion")
            candidates = [("calibrada", chosen, start_res, self._evaluate("calibrada"))]
            candidates.append(("referencia", start_settings, start_res, baseline))

            for width, height in RESOLUTIONS:
                self._check_cancel()
                if (width, height) == start_res:
                    continue
                self._say(f"Probando {width}x{height}...", 80, "resolucion")
                if not self.camera.switch(self.camera.index, width, height):
                    self._say(f"{width}x{height} no disponible")
                    continue
                self.camera.apply(chosen)
                candidates.append(
                    (
                        f"{width}x{height}",
                        chosen,
                        (width, height),
                        self._evaluate(f"{width}x{height}"),
                    )
                )

            best = self._pick(candidates)
            self._finish(best, candidates, on_done)
        except _Cancelled:
            self.camera.switch(self.camera.index, *start_res)
            self.camera.apply(start_settings)
            self.state.status = "cancelled"
            self._say("Calibracion cancelada; se restauraron los ajustes iniciales")
        except Exception as exc:
            log.exception("calibracion fallida")
            self.state.status = "error"
            self._say(f"Error: {type(exc).__name__}")
        finally:
            self._persist()

    def _persist(self) -> None:
        """Guarda el log y el resultado (sin datos de documentos) para revisarlos despues."""
        if self._log_dir is None:
            return
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            path = self._log_dir / f"calibracion_{datetime.now():%Y%m%d_%H%M%S}.json"
            payload = {
                "estado": self.state.status,
                "log": self.state.log,
                "resultado": self.state.result,
            }
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            log.warning("no se pudo guardar el log de calibracion", exc_info=True)

    @staticmethod
    def _pick(candidates):
        """Gana quien lee mas veces la MRZ. Entre iguales, solo se cambia la referencia si la otra
        es claramente mas nitida (la nitidez fluctua por ruido: una diferencia chica no cuenta)."""
        reference = next(c for c in candidates if c[0] == "referencia")
        top_reads = max(c[3]["lecturas_ok"] for c in candidates)
        contenders = [c for c in candidates if c[3]["lecturas_ok"] == top_reads]
        if reference in contenders:
            sharpest = max(contenders, key=lambda c: c[3]["sharp"])
            if sharpest[3]["sharp"] <= reference[3]["sharp"] * MIN_SHARPNESS_GAIN:
                return reference
            return sharpest
        return max(contenders, key=lambda c: c[3]["sharp"])

    def _finish(self, best, candidates, on_done) -> None:
        label, settings, resolution, evaluation = best
        self.camera.switch(self.camera.index, *resolution)
        self.camera.apply(settings)
        reads, total = evaluation["lecturas_ok"], evaluation["lecturas"]
        validated = total > 0 and reads == total  # 2/3 NO es validada: una captura suelta falla
        stability = "estable" if validated else ("fragil" if reads > 0 else "sin_lectura")
        if stability == "fragil":
            self._say(
                f"AVISO: la mejor configuracion solo leyo {reads}/{total} veces: es FRAGIL. Una "
                "lectura suelta falla a menudo. Mejora la luz, la distancia o el enfoque."
            )
        if not validated and stability == "sin_lectura":
            self._say(
                "AVISO: ninguna configuracion leyo la MRZ. Pon un documento de prueba con MRZ "
                "quieto frente a la camara y repite; sin el solo se optimizo por nitidez y "
                "brillo y no se pudo validar."
            )
        result = {
            "ganadora": label,
            "ajustes": settings,
            "resolucion": list(resolution),
            "evaluacion": evaluation,
            "validada_con_mrz": validated,
            "estabilidad": stability,
            "candidatas": [c[3] for c in candidates],
        }
        self.state.result = result
        self.state.status = "done"
        self._say(f"Resultado: '{label}'", 100, "listo")
        if on_done:
            on_done(result)


class _Cancelled(Exception):
    pass
