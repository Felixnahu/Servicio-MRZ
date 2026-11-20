"""Reconocimiento continuo: desde que entra un documento intenta leerlo y entrega el JSON.

Fases:
  buscando  -> intenta leer la MRZ con el ultimo cuadro, una y otra vez
  leido     -> ya hay resultado; se mantiene mientras el documento siga ahi
  pausado   -> calibracion en curso o sin camara

Un documento = un solo evento `resultado`. Cuando el documento se retira (la escena cambia de
forma sostenida) se emite `retirado` y se vuelve a `buscando`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from servicio_mrz import contract
from servicio_mrz.fusion import Fuser

log = logging.getLogger(__name__)

SIGNATURE_SIZE = (64, 36)
REMOVED_DIFF = 12.0  # diferencia media (0-255) que indica que la escena cambio
REMOVED_POLLS = (
    3  # sondeos consecutivos sobre el umbral (~0.6 s) para dar el documento por retirado
)
POLL_S = 0.2
NO_MRZ_AFTER = 3  # intentos fallidos con texto visible antes de sugerir "dar la vuelta"
NO_MRZ_MIN_BOXES = 4
IDLE_CHANGE = 6.0  # con la escena vacia, solo se vuelve a intentar si cambia al menos esto...
ATTEMPT_REST_RATIO = (
    1.0  # tras cada intento se descansa esto x lo que tardo (50 % de CPU como mucho)
)
ATTEMPT_MIN_GAP_S = 0.25
IDLE_RETRY_S = 8.0  # ...o pasado este tiempo (un documento claro sobre fondo claro cambia poco)


def signature(frame: np.ndarray) -> np.ndarray:
    """Miniatura gris y suavizada: barata y estable frente al ruido del sensor."""
    small = cv2.resize(frame, SIGNATURE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)


def scene_change(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


class Recognizer:
    def __init__(
        self,
        camera: Any,
        reader: Any,
        publish: Callable[[dict[str, Any]], None],
        paused: Callable[[], bool] = lambda: False,
    ) -> None:
        self.camera, self.reader, self._publish, self._paused = camera, reader, publish, paused
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._fuser = Fuser()
        self._state: dict[str, Any] = self._fresh("buscando")

    @staticmethod
    def _fresh(phase: str) -> dict[str, Any]:
        return {
            "fase": phase,
            "intentos": 0,
            "ultimo_intento_ms": None,
            "filas": [],
            "controles": None,
            "formato": None,
            "cajas": 0,
            "tarjeta": None,
            "cambio_escena": None,
            "resultado": None,
            "pista": None,
            "fusionados": 0,
        }

    # --- ciclo de vida -------------------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="recognizer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _set(self, **changes: Any) -> None:
        with self._lock:
            self._state.update(changes)

    def _emit(self, kind: str, **data: Any) -> None:
        try:
            self._publish({"tipo": kind, **data})
        except Exception:
            log.exception("no se pudo publicar el evento %s", kind)

    # --- bucle ---------------------------------------------------------------------------
    def _loop(self) -> None:
        sig_at_read: np.ndarray | None = None
        far_polls = 0
        failures = 0
        idle_sig: np.ndarray | None = None  # ultima escena en la que no habia nada
        idle_since = 0.0
        rest_until = 0.0  # no se reintenta antes de este instante (limita el uso de CPU)
        while not self._stop.is_set():
            if self._paused() or not self.camera.running:
                if self.snapshot()["fase"] != "pausado":
                    self._set(fase="pausado")
                    self._emit("estado", **self.snapshot())
                time.sleep(POLL_S)
                continue
            if self.snapshot()["fase"] == "pausado":
                self._fuser.reset()
                self._set(**self._fresh("buscando"))
                sig_at_read, far_polls, failures, idle_sig = None, 0, 0, None
                self._emit("estado", **self.snapshot())

            latest = self.camera.latest()
            if latest is None:
                time.sleep(POLL_S)
                continue
            frame = latest[1]

            if self.snapshot()["fase"] == "leido":
                change = (
                    scene_change(sig_at_read, signature(frame)) if sig_at_read is not None else 0
                )
                far_polls = far_polls + 1 if change > REMOVED_DIFF else 0
                self._set(cambio_escena=round(change, 1))
                if far_polls >= REMOVED_POLLS:
                    self._fuser.reset()
                    self._set(**self._fresh("buscando"))
                    sig_at_read, far_polls, failures, idle_sig = None, 0, 0, None
                    self._emit("retirado")
                    self._emit("estado", **self.snapshot())
                time.sleep(POLL_S)
                continue

            if idle_sig is not None:
                moved = scene_change(idle_sig, signature(frame)) >= IDLE_CHANGE
                if not moved and time.monotonic() - idle_since < IDLE_RETRY_S:
                    time.sleep(POLL_S)
                    continue

            if time.monotonic() < rest_until:
                time.sleep(POLL_S)
                continue
            attempt_sig = signature(frame)
            reading = self.reader.read_detailed(frame)
            rest_until = time.monotonic() + max(
                ATTEMPT_MIN_GAP_S, reading.ms / 1000 * ATTEMPT_REST_RATIO
            )
            attempts = self.snapshot()["intentos"] + 1
            result = reading.result if reading.checks_ok else None
            if result is None and reading.observation is not None:
                # un cuadro suelto no valido: sumarlo a los del mismo documento y probar la fusion
                self._fuser.add(reading.observation)
                result = self._fuser.fuse()
            rows = [
                {"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1, "chars": r.chars, "usada": r.used}
                for r in reading.rows
            ]
            card = [{"x": x, "y": y} for x, y in reading.card] if reading.card else None
            shown = result or reading.result
            info = {
                "intentos": attempts,
                "ultimo_intento_ms": reading.ms,
                "filas": rows,
                "tarjeta": card,
                "cajas": reading.boxes,
                "formato": shown.format if shown else None,
                "controles": shown.checks if shown else None,
                "fusionados": self._fuser.size,
            }
            if result is not None:
                payload = contract.from_mrz(result, elapsed_ms=reading.ms)
                sig_at_read, far_polls, failures = signature(frame), 0, 0
                self._set(fase="leido", resultado=payload, pista=None, cambio_escena=0.0, **info)
                log.info(
                    "documento leido en %s ms (intento %s, %s cuadros fusionados)",
                    reading.ms,
                    attempts,
                    result.fused_frames,
                )
                self._emit("resultado", resultado=payload)
                self._emit("estado", **self.snapshot())
                continue

            failures += 1
            nothing_there = reading.card is None  # sin tarjeta: esperar a que la escena cambie
            idle_sig, idle_since = (attempt_sig, time.monotonic()) if nothing_there else (None, 0.0)
            card_without_mrz = reading.card is not None and not rows
            text_without_mrz = (
                reading.card is None and not rows and reading.boxes >= NO_MRZ_MIN_BOXES
            )
            hint = None
            if failures >= NO_MRZ_AFTER and (card_without_mrz or text_without_mrz):
                hint = "hay un documento pero ninguna MRZ: pruebe la otra cara del documento"
                if failures == NO_MRZ_AFTER:
                    self._emit("sin_mrz", mensaje=hint)
            self._set(pista=hint, **info)
            self._emit("estado", **self.snapshot())
