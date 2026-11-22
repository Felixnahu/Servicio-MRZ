"""Lectura de la zona MRZ desde una imagen: OCR (RapidOCR/ONNX, solo CPU) + parser con checksums.

El OCR no se fia: lo que decide es que los digitos de control cuadren. Se prueban varias
candidatas de imagen (completa, franja inferior ampliada) y gana la que mas controles pasa.
Ademas de los datos, devuelve la geometria de lo detectado (para dibujarlo y depurar).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from servicio_mrz import birthplace, cardfinder, celldecode, mrzread, oldci, orientation, visual
from servicio_mrz.mrz import MrzResult, parse_mrz
from servicio_mrz.mrzread import Observation
from servicio_mrz.textclean import clean_row

log = logging.getLogger("servicio_mrz")

_MAX_SIDE = 1600
_MIN_ROW = 20  # largo minimo para intentar parsear una fila
_MIN_DISPLAY = 15  # largo minimo para dibujarla como "fila tipo MRZ"


@dataclass(frozen=True)
class _Box:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class _Row:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class RowBox:
    """Fila de texto tipo MRZ, en coordenadas normalizadas (0..1) de la imagen original."""

    x0: float
    y0: float
    x1: float
    y1: float
    chars: int
    used: bool  # forma parte de la ventana que se parseo


@dataclass(frozen=True)
class Reading:
    result: MrzResult | None
    rows: tuple[RowBox, ...] = ()
    boxes: int = 0  # cajas de texto totales (hay "algo con texto" en la imagen)
    ms: int = 0
    card: tuple[tuple[float, float], ...] | None = None  # 4 esquinas normalizadas, si se localizo
    observation: Observation | None = None  # probabilidades por celda, para fusionar cuadros
    card_aspect: float = 0.0  # proporcion lado largo/corto de la tarjeta localizada (0 = ninguna)
    mrz_confidence: float = 0.0  # mejor confianza de la lectura de MRZ (0..1); solo para el log
    checks_ok: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks_ok", bool(self.result and self.result.core_checks_ok))


def _rows(boxes: list[_Box]) -> list[_Row]:
    groups: list[list[_Box]] = []
    for box in sorted(boxes, key=lambda b: b.cy):
        if groups and abs(groups[-1][-1].cy - box.cy) < 0.6 * max(box.h, groups[-1][-1].h):
            groups[-1].append(box)
        else:
            groups.append([box])
    rows = []
    for g in groups:
        g.sort(key=lambda b: b.x0)
        rows.append(
            _Row(
                "".join(clean_row(b.text) for b in g),
                min(b.x0 for b in g),
                min(b.y0 for b in g),
                max(b.x1 for b in g),
                max(b.y1 for b in g),
            )
        )
    return rows


def _score(result: MrzResult) -> int:
    return sum(result.checks.values())


def _looks_mrz(text: str) -> bool:
    if len(text) < _MIN_DISPLAY:
        return False
    digits = sum(c.isdigit() for c in text) / len(text)
    return "<" in text or digits >= 0.3


def best_window(rows: list[_Row]) -> tuple[MrzResult, list[_Row]] | None:
    """Ventana de 2 o 3 filas consecutivas con mas digitos de control correctos."""
    usable = [r for r in rows if len(r.text) >= _MIN_ROW]
    best: tuple[MrzResult, list[_Row]] | None = None
    for count in (3, 2):
        for i in range(len(usable) - count + 1):
            window = usable[i : i + count]
            parsed = parse_mrz([r.text for r in window])
            if parsed and (best is None or _score(parsed) > _score(best[0])):
                best = (parsed, window)
    return best


def best_mrz(rows: list[str]) -> MrzResult | None:
    found = best_window([_Row(t, 0, 0, 0, 0) for t in rows])
    return found[0] if found else None


def _resize(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = _MAX_SIDE / max(h, w)
    if scale >= 1:
        return img, 1.0
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def _candidates(img: np.ndarray):
    """(imagen, escala, desplazamiento_y) respecto a la ORIGINAL: franja inferior y luego completa.

    La franja (donde esta la MRZ) cuesta menos y, medido, ya lee todo lo que lee la imagen completa.
    """
    resized, r = _resize(img)
    h = resized.shape[0]
    if resized.shape[1] / h > _STRIP_ASPECT:  # recorte ya apaisado: es la propia franja MRZ
        yield resized, r, 0.0
        return
    top = int(h * 0.55)
    strip = resized[top:, :]
    up = 1.0
    if strip.shape[1] < _MAX_SIDE:
        up = 1.6
        strip = cv2.resize(strip, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    yield strip, r * up, top / r
    yield resized, r, 0.0


def _segment_box(seg, card_shape, rotated, to_frame, frame_size, chars, used) -> RowBox:
    """Caja (normalizada, en el cuadro original) de una linea hallada en la tarjeta enderezada."""
    ch, cw = card_shape[:2]
    xs, ys = np.array([seg.x0, seg.x1, seg.x1, seg.x0]), np.array([seg.y0, seg.y0, seg.y1, seg.y1])
    if rotated:  # las coordenadas estan en la tarjeta girada 180: volver a la enderezada
        xs, ys = cw - 1 - xs, ch - 1 - ys
    pts = cardfinder.project(np.stack([xs, ys], axis=1), to_frame)
    width, height = frame_size
    return RowBox(
        float(pts[:, 0].min()) / width,
        float(pts[:, 1].min()) / height,
        float(pts[:, 0].max()) / width,
        float(pts[:, 1].max()) / height,
        chars=chars,
        used=used,
    )


# Tope de analisis por foto. Una operacion en curso no se interrumpe: se pasa hasta ~0.3 s.
DEFAULT_BUDGET_S = 2.7
_STRIP_ASPECT = 3.0  # ancho/alto a partir del cual la imagen se considera ya una franja MRZ
_NEXT_CANDIDATE_COST = 1.3  # el siguiente candidato cuesta ~1.3x el anterior (medido)


def _enable_memory_arena() -> None:
    """RapidOCR desactiva el arena de memoria de ONNX (`enable_cpu_mem_arena = False`).

    Con formas de entrada que cambian en cada foto (anchos de linea y de caja distintos) cada
    inferencia pide memoria nueva al sistema: medido, una tarjeta de carnet tarda 2.8 s sin arena y
    0.8 s con ella (3.4x), con los mismos resultados. Se activa antes de crear las sesiones.
    """
    from rapidocr_onnxruntime.utils import infer_engine

    original = infer_engine.OrtInferSession._init_sess_opts
    if getattr(original, "_arena_patched", False):
        return

    def with_arena(config):
        options = original(config)
        options.enable_cpu_mem_arena = True
        return options

    with_arena._arena_patched = True  # type: ignore[attr-defined]
    infer_engine.OrtInferSession._init_sess_opts = staticmethod(with_arena)


def make_engine(threads: int = 0, mem_arena: bool = True):
    """Motor RapidOCR. `threads` > 0 limita los hilos de ONNX (0 = todos los nucleos).

    Medido (6 fotos): 1 hilo 2.05 s/foto con ~1.2 nucleos; 2 hilos 1.66 s con ~2.4; 4 hilos 1.20 s
    con ~4.7; 8 hilos 0.95 s con ~9.5. Mas hilos ganan poco tiempo y gastan mucha mas CPU.
    """
    from rapidocr_onnxruntime import RapidOCR

    if mem_arena:
        _enable_memory_arena()
    if threads <= 0:
        return RapidOCR()
    import tempfile

    import yaml
    from rapidocr_onnxruntime.main import DEFAULT_CFG_PATH, read_yaml

    config = read_yaml(DEFAULT_CFG_PATH)
    for section in ("Det", "Cls", "Rec"):
        config[section]["intra_op_num_threads"] = threads
        config[section]["inter_op_num_threads"] = 1
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    try:
        return RapidOCR(config_path=f.name)
    finally:
        Path(f.name).unlink(missing_ok=True)


class MrzReader:
    def __init__(
        self, threads: int = 0, budget_s: float = DEFAULT_BUDGET_S, mem_arena: bool = True
    ) -> None:
        self._budget_s = budget_s
        self._mem_arena = mem_arena
        self._engine = None
        self._line_recognizer = None
        self._init_lock = threading.Lock()
        self._threads = threads
        if threads > 0:
            cv2.setNumThreads(threads)  # tambien el modelo de esquinas (cv2.dnn)

    def _ocr(self, img: np.ndarray) -> list[_Box]:
        result, _ = self._engine_ready()(img)
        boxes = []
        for points, text, _conf in result or []:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            boxes.append(_Box(min(xs), min(ys), max(xs), max(ys), text))
        return boxes

    def warm_up(self) -> None:
        """Carga los modelos y hace una inferencia de prueba: la primera foto real no la paga."""
        started = time.monotonic()
        try:
            blank = np.full((480, 720, 3), 200, np.uint8)
            self.read_detailed(blank, budget_s=1.0)
            self.read_visual(blank)
            log.info("Modelos listos en %.1f s", time.monotonic() - started)
        except Exception:  # noqa: BLE001 - el calentamiento nunca debe tumbar el servicio
            log.warning("No se pudo calentar los modelos (se cargaran con la primera foto)")

    def read_old_ci(
        self, images: list[np.ndarray], deadline: float | None = None, detect_student: bool = True
    ) -> dict | None:
        """Cedula boliviana antigua (sin MRZ) por plantilla: nombre y fecha del reverso, numero del
        anverso. Si es un carnet universitario, solo la categoria. None si no se lee la fecha de
        nacimiento o se agota `deadline`."""
        return oldci.read_old_ci(self._engine_ready(), images, deadline, detect_student)

    def read_birthplace(self, img: np.ndarray) -> bool | None:
        """Cedula 2023 (reverso): True/False/None segun si el lugar de nacimiento es Samaipata.

        Solo lee la franja superior de la tarjeta, donde va "LUGAR DE NACIMIENTO" (~0.3 s).
        """
        engine = self._engine_ready()
        card = visual._prepare(img)
        for rot in (card, cv2.rotate(card, cv2.ROTATE_180)):
            band = rot[: int(0.3 * rot.shape[0])]
            boxes, width, _ = visual.read_boxes(engine, band, use_cls=False)
            if any(birthplace._is_label(b.text) for b in boxes):
                return birthplace.from_new_boxes(boxes, width)
        return None

    def read_visual(self, img: np.ndarray) -> dict | None:
        """Campos IMPRESOS (OCR por etiqueta): respaldo cuando no hay MRZ legible (~1.7 s)."""
        return visual.read_image(self._engine_ready(), img)

    def read(self, img: np.ndarray) -> MrzResult | None:
        return self.read_detailed(img).result

    def read_detailed(self, img: np.ndarray, budget_s: float | None = None) -> Reading:
        """Mejor MRZ hallada y la geometria de lo detectado (para dibujar y depurar).

        Camino rapido: localizar la tarjeta, enderezarla y reconocer sus lineas MRZ una a una.
        Respaldo: OCR general sobre el cuadro (tarjeta no localizada, p. ej. pasaporte en la mano).
        """
        started = time.monotonic()
        deadline = started + (self._budget_s if budget_s is None else budget_s)
        card_reading = self._read_card(img, started, deadline)
        if card_reading is not None:
            # Con la tarjeta localizada el camino por lineas es el bueno; el OCR general (4-8 s por
            # cuadro) solo se usa cuando NO se localiza la tarjeta (p. ej. pasaporte en la mano).
            return card_reading
        return self._read_frame(img, started, deadline)

    def _engine_ready(self):
        with self._init_lock:  # la carga perezosa puede ocurrir desde varios hilos
            if self._engine is None:
                self._engine = make_engine(self._threads, self._mem_arena)
        return self._engine

    def _read_card(self, img: np.ndarray, started: float, deadline: float) -> Reading | None:
        """None si no se localiza ninguna tarjeta."""
        frame_flipped = False
        try:
            located = cardfinder.locate_card(img)
        except Exception:
            located = None
        if located is None:
            # El localizador no es invariante al giro (medido: pierde ~5 % de las fotos al revez y
            # halla otro ~3 % solo girado). Se prueba el cuadro girado 180 grados antes de rendirse.
            upside = cv2.rotate(img, cv2.ROTATE_180)
            try:
                again = cardfinder.locate_card(upside)
            except Exception:
                again = None
            if again is not None:
                img, located, frame_flipped = upside, again, True
        height, width = img.shape[:2]
        if located is not None:
            warped = cardfinder.warp_card(img, located)
        else:
            # un recorte ya enderezado (la app puede mandar la tarjeta ajustada) se lee entero
            warped = cardfinder.whole_image_as_card(img)
            if warped is None:
                return None
            located = warped.located
        engine = self._engine_ready()
        if self._line_recognizer is None:
            self._line_recognizer = celldecode.LineRecognizer(engine)
        # La tarjeta puede estar al reves (0 / 180 grados): se empieza por el sentido que el
        # clasificador de giro considera correcto; el otro queda como segunda pasada.
        flipped, firmness = orientation.assess(engine, warped.image)
        if flipped is None and orientation.sideways(engine, warped.image):
            # Tarjeta puesta de lado: casi todo su texto es vertical y la MRZ solo se lee en
            # horizontal, asi que las dos lecturas (0 y 180 grados) gastaban 10-17 s en el totem
            # sin poder acertar. Sigue el lector de carnet antiguo/universitario, que la endereza.
            side_norm = tuple((float(x) / width, float(y) / height) for x, y in located.corners)
            elapsed = int((time.monotonic() - started) * 1000)
            return Reading(None, (), 0, elapsed, side_norm, None, located.aspect)
        if flipped:
            # El localizador recorta mejor una tarjeta del derecho: girar la FOTO y volver a
            # localizar suele dar un enderezado mas limpio que girar la tarjeta ya enderezada.
            upside = cv2.rotate(img, cv2.ROTATE_180)
            try:
                again = cardfinder.locate_card(upside)
            except Exception:
                again = None
            if again is not None:
                retry = cardfinder.warp_card(upside, again)
                retry_flipped, retry_firmness = orientation.assess(engine, retry.image)
                if retry_flipped is False:
                    img, located, warped = upside, again, retry
                    frame_flipped, flipped, firmness = not frame_flipped, False, retry_firmness
        card_norm = tuple((float(x) / width, float(y) / height) for x, y in located.corners)
        best: MrzResult | None = None
        best_rows: tuple[RowBox, ...] = ()
        observation: Observation | None = None
        best_confidence = 0.0
        # Voto firme: solo ese sentido (la otra pasada cuesta de 3 a 5 s en el totem y casi nunca
        # sirve). Voto dudoso o sin voto: los dos, empezando por el mas probable.
        if flipped is not None and firmness >= orientation.DECISIVE_RATIO:
            order = (bool(flipped),)
        else:
            order = (True, False) if flipped else (False, True)
        for index, rotated in enumerate(order):
            if index and time.monotonic() >= deadline:
                break
            card = cv2.rotate(warped.image, cv2.ROTATE_180) if rotated else warped.image
            read = mrzread.read_card(
                engine,
                self._line_recognizer,
                card,
                prefer_lines=2 if located.aspect < 1.5 else 3,
                deadline=deadline,
            )
            if read.observation and (
                observation is None or read.observation.confidence > observation.confidence
            ):
                observation = read.observation
            best_confidence = max(best_confidence, float(read.confidence))
            if len(read.segments) < 2:
                continue
            used = bool(read.result and read.result.core_checks_ok)
            rows = tuple(
                _segment_box(seg, card.shape, rotated, warped.to_frame, (width, height), 0, used)
                for seg in read.segments
            )
            if read.result and (best is None or _score(read.result) > _score(best)):
                best, best_rows = read.result, rows
            elif not best_rows:
                best_rows = rows
            if best and best.core_checks_ok:
                break
            if read.observation is not None and index == 0:
                break  # texto reconocido con confianza en el primer sentido: no probar el otro
        if frame_flipped:  # volver al cuadro original (las cajas y esquinas son para dibujar)
            card_norm = tuple((1.0 - x, 1.0 - y) for x, y in card_norm)
            best_rows = tuple(
                RowBox(1.0 - r.x1, 1.0 - r.y1, 1.0 - r.x0, 1.0 - r.y0, r.chars, r.used)
                for r in best_rows
            )
        elapsed = int((time.monotonic() - started) * 1000)
        return Reading(
            best, best_rows, 0, elapsed, card_norm, observation, located.aspect, best_confidence
        )

    def _read_frame(self, img: np.ndarray, started: float, deadline: float) -> Reading:
        height, width = img.shape[:2]
        best: MrzResult | None = None
        best_rows: tuple[RowBox, ...] = ()
        total_boxes = 0

        last_cost = 0.0
        for cand, scale, y_off in _candidates(img):
            if last_cost and time.monotonic() + last_cost * _NEXT_CANDIDATE_COST > deadline:
                break  # no queda tiempo para otro OCR completo
            t = time.monotonic()
            boxes = self._ocr(cand)
            last_cost = time.monotonic() - t
            total_boxes = max(total_boxes, len(boxes))
            rows = _rows(boxes)
            found = best_window(rows)
            used = {id(r) for r in found[1]} if found else set()

            def norm(r: _Row, scale: float = scale, y_off: float = y_off) -> tuple[float, ...]:
                return (
                    r.x0 / scale / width,
                    (y_off + r.y0 / scale) / height,
                    r.x1 / scale / width,
                    (y_off + r.y1 / scale) / height,
                )

            display = tuple(
                RowBox(*norm(r), chars=len(r.text), used=id(r) in used)
                for r in rows
                if _looks_mrz(r.text)
            )
            better = found and (best is None or _score(found[0]) > _score(best))
            if better or (best is None and display and not best_rows):
                if found:
                    best = found[0]
                best_rows = display
            if best and best.core_checks_ok:
                break
        return Reading(best, best_rows, total_boxes, int((time.monotonic() - started) * 1000))
