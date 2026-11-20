"""Camara web duena del servicio: un hilo captura y deja siempre el ultimo cuadro disponible.

Ni la app ni otros modulos abren el dispositivo: todos piden el cuadro a esta clase. Los ajustes
(brillo, contraste, enfoque, exposicion) se aplican con `apply()`; que una webcam los acepte
depende del driver, por eso `apply()` informa cuales aceptó.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque

import cv2
import numpy as np

log = logging.getLogger(__name__)

MIN_FRAME_GAP_S = 0.004

PROPS = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "focus": cv2.CAP_PROP_FOCUS,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
}


def device_names() -> list[str]:
    """Nombres de las camaras de Windows, en el mismo orden que los indices de DirectShow.

    Enumerar no abre los dispositivos. Si la libreria falla, devuelve [] y la UI usa 'Camara N'.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(FilterGraph().get_input_devices())
    except Exception:
        log.warning("No se pudieron listar los nombres de camara", exc_info=True)
        return []


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.index, self.width, self.height = index, width, height
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stamps: deque[float] = deque(maxlen=31)

    # --- ciclo de vida -------------------------------------------------------------------
    def start(self) -> bool:
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            log.error("No se pudo abrir la camara %s", self.index)
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
        self._cap = self._thread = None
        with self._lock:
            self._frame = None

    def switch(self, index: int, width: int | None = None, height: int | None = None) -> bool:
        """Cambia de camara y/o resolucion. Si no abre, intenta volver a la anterior."""
        previous = (self.index, self.width, self.height)
        self.stop()
        self.index = index
        self.width, self.height = width or self.width, height or self.height
        if self.start():
            return True
        self.index, self.width, self.height = previous
        self.start()
        return False

    def _loop(self) -> None:
        assert self._cap is not None
        self._stamps.clear()
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            self._stamps.append(time.monotonic())
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def fps(self) -> float:
        """Cuadros por segundo recientes (con exposicion larga la camara baja de ritmo).

        Mediana de los ultimos intervalos: robusta a las rafagas de cuadros acumulados que algunos
        drivers entregan al abrir (un promedio movil daba cientos de miles de fps).
        """
        stamps = list(self._stamps)
        # los intervalos < 4 ms (> 250 fps) son rafagas de cuadros acumulados, no el ritmo real
        gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False) if b - a > MIN_FRAME_GAP_S]
        return 1.0 / statistics.median(gaps) if gaps else 0.0

    # --- lectura -------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest(self) -> tuple[int, np.ndarray] | None:
        """(id del cuadro, copia del ultimo cuadro) o None si aun no llego ninguno."""
        with self._lock:
            return None if self._frame is None else (self._frame_id, self._frame.copy())

    def resolution(self) -> tuple[int, int] | None:
        if not self._cap:
            return None
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    def grab(self, count: int = 3, discard: int = 2, timeout: float = 4.0) -> list[np.ndarray]:
        """Cuadros NUEVOS tras un cambio de ajuste (los primeros aun traen el valor viejo)."""
        start = self.latest()
        first_id = start[0] if start else 0
        frames: list[np.ndarray] = []
        last = first_id
        deadline = time.monotonic() + timeout
        while len(frames) < count and time.monotonic() < deadline:
            latest = self.latest()
            if latest and latest[0] > last:
                last = latest[0]
                if last > first_id + discard:
                    frames.append(latest[1])
            else:
                time.sleep(0.01)
        return frames

    def grab_settled(
        self, count: int = 3, min_wait: int = 8, timeout: float = 6.0
    ) -> list[np.ndarray]:
        """Cuadros ya ASENTADOS tras cambiar un ajuste.

        Medido en una webcam real: el efecto de un cambio de exposicion tarda ~5 cuadros en
        verse (y con exposicion larga la camara baja a ~6 fps). Por eso se espera un minimo de
        cuadros y despues a que el brillo se estabilice. Si no se estabiliza en `timeout`, se
        devuelven los ultimos cuadros (mejor eso que nada).
        """
        last_id = (self.latest() or (0, None))[0]
        seen, history, run = 0, [], []
        recent: list[np.ndarray] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            latest = self.latest()
            if not latest or latest[0] <= last_id:
                time.sleep(0.01)
                continue
            last_id, frame = latest
            seen += 1
            brightness = float(frame.mean())
            history.append(brightness)
            recent = (recent + [frame])[-count:]
            window = history[-3:]
            tolerance = max(2.0, 0.03 * brightness)
            stable = (
                seen >= min_wait and len(window) == 3 and max(window) - min(window) <= tolerance
            )
            run = run + [frame] if stable else []
            if len(run) >= count:
                return run[-count:]
        return recent

    def get(self, name: str) -> float | None:
        prop = PROPS.get(name)
        return float(self._cap.get(prop)) if self._cap is not None and prop is not None else None

    # --- ajustes -------------------------------------------------------------------------
    def apply(self, settings: dict[str, float]) -> dict[str, bool]:
        """Aplica ajustes; devuelve cuales acepto el driver (lectura posterior coincide)."""
        accepted: dict[str, bool] = {}
        if not self._cap:
            return accepted
        for name, value in settings.items():
            prop = PROPS.get(name)
            if prop is None:
                accepted[name] = False
                continue
            self._cap.set(prop, float(value))
            accepted[name] = abs(self._cap.get(prop) - float(value)) < 1e-6
        return accepted


def encode_jpeg(frame: np.ndarray, quality: int = 80, max_width: int | None = None) -> bytes:
    if max_width and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("no se pudo codificar el cuadro")
    return buf.tobytes()
