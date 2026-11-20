"""Lectura de la MRZ sobre la tarjeta ya enderezada: segmenta las lineas de texto de la franja
inferior y reconoce cada una por separado (sin deteccion de texto).

Por que asi: el OCR general detecta y reconoce toda la franja a la vez y se confunde con los '<'
(pierde o fusiona lineas). Con la tarjeta enderezada las lineas MRZ caen siempre en la misma zona,
son 2 o 3 y miden casi todo el ancho; reconocerlas una a una es mas rapido y mas fiable.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BAND_START = 0.55  # la MRZ vive en el ~45 % inferior de la tarjeta
_BORDER_ROW_COVERAGE = 0.75
_BLACKHAT_KERNEL = (35, 35)
_LINE_HEIGHT_PX = (14, 80)  # alto plausible de una linea en la tarjeta de 1024 px de ancho
_MIN_LINE_WIDTH = 0.55  # una linea MRZ cubre mas de la mitad del ancho de la tarjeta


@dataclass(frozen=True)
class LineSegment:
    y0: int
    y1: int
    x0: int
    x1: int


def segment_lines(card: np.ndarray, drop_border_rows: bool = True) -> list[LineSegment]:
    """Lineas de texto anchas de la franja inferior, de arriba abajo."""
    h, w = card.shape[:2]
    y_start = int(h * BAND_START)
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    # el blackhat resalta texto oscuro sobre fondo claro aunque haya destellos o degradados
    black = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, _BLACKHAT_KERNEL)
    )
    _, ink = cv2.threshold(black, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink[:, : int(w * 0.04)] = 0
    ink[:, int(w * 0.985) :] = 0
    # el borde de la tarjeta (o el fondo oscuro tras enderezar) es una fila con tinta de lado a
    # lado; el texto real cubre ~30 % del ancho. Sin descartarla dispara el maximo del perfil y
    # el umbral relativo elimina las lineas reales.
    if drop_border_rows:
        ink[ink.sum(axis=1) / 255.0 > w * _BORDER_ROW_COVERAGE] = 0
    profile = np.convolve(ink[y_start:].sum(axis=1) / 255.0, np.ones(7) / 7, mode="same")
    on = profile > max(profile.max() * 0.30, 20.0)

    segments: list[LineSegment] = []
    y = 0
    while y < len(on):
        if not on[y]:
            y += 1
            continue
        top = y
        while y < len(on) and on[y]:
            y += 1
        if not _LINE_HEIGHT_PX[0] <= y - top <= _LINE_HEIGHT_PX[1]:
            continue
        y0, y1 = top + y_start, y + y_start
        cols = np.nonzero(ink[y0:y1].sum(axis=0) > 255 * 2)[0]
        if len(cols) and (cols.max() - cols.min()) > w * _MIN_LINE_WIDTH:
            segments.append(LineSegment(y0, y1, int(cols.min()), int(cols.max())))
    return segments
