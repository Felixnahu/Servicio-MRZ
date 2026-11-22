"""Sentido de una tarjeta ya enderezada: del derecho o al reves (giro de 180 grados).

El localizador de tarjetas devuelve siempre una imagen apaisada, pero el texto puede quedar al
reves (el visitante metio el carnet girado). Hasta ahora se leia primero en un sentido y, si
fallaba, se repetia TODA la lectura en el otro: en el totem cada pasada cuesta de 4 a 9 s.

Aqui se decide antes con el clasificador de giro del propio OCR: detecta las lineas de texto y vota
si cada una esta al derecho o al reves. Cuesta ~0.1 s en un portatil (una fraccion de una
lectura) y se midio sobre 298 fotos reales: en las 290 con texto suficiente el voto de la tarjeta
y el de su giro de 180 grados salieron SIEMPRE opuestos. Solo cambia el ORDEN de las pasadas: si
el voto se equivoca, la otra pasada sigue ahi como antes.

Usa piezas internas de RapidOCR (`text_det`, `get_crop_img_list`, `text_cls`): si cambian o fallan,
devuelve None y el llamador conserva el orden de siempre.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

MIN_LINES = 3  # lineas horizontales largas necesarias para votar
MIN_RATIO = 0.2  # |voto| / peso total: por debajo es un empate y no se decide
_LONG = 2.0  # una linea de texto es al menos 2 veces mas ancha que alta
_MIN_WIDTH = 60.0


def _votes(engine: Any, card: np.ndarray) -> tuple[float, float, int]:
    """(voto neto, peso total, lineas): >0 si el texto va del derecho."""
    boxes, _ = engine.text_det(card)
    if boxes is None or len(boxes) == 0:
        return 0.0, 0.0, 0
    lines = []
    for box in boxes:
        width = float(np.linalg.norm(box[0] - box[1]))
        height = float(np.linalg.norm(box[0] - box[3]))
        if width >= _LONG * height and width >= _MIN_WIDTH:
            lines.append((np.asarray(box), width))
    if not lines:
        return 0.0, 0.0, 0
    crops = engine.get_crop_img_list(card, np.array([b for b, _ in lines]))
    _, labels, _ = engine.text_cls(crops)
    net = total = 0.0
    for (label, confidence), (_, width) in zip(labels, lines, strict=True):
        weight = width * float(confidence)
        total += weight
        net += weight if label == "0" else -weight
    return net, total, len(lines)


def upside_down(engine: Any, card: np.ndarray) -> bool | None:
    """True si el texto de la tarjeta esta al reves, False si va del derecho, None si no se sabe."""
    try:
        net, total, lines = _votes(engine, card)
    except Exception:  # noqa: BLE001 - es una optimizacion: nunca debe tumbar la lectura
        log.warning("No se pudo estimar el sentido de la tarjeta", exc_info=True)
        return None
    if lines < MIN_LINES or total <= 0 or abs(net) / total < MIN_RATIO:
        return None
    return net < 0


def sideways(engine: Any, card: np.ndarray) -> bool:
    """True si casi todo el texto detectado va en vertical (tarjeta puesta de lado en la ranura)."""
    try:
        boxes, _ = engine.text_det(card)
        if boxes is None:
            return False
        vertical = horizontal = 0
        for box in boxes:
            width = float(np.linalg.norm(box[0] - box[1]))
            height = float(np.linalg.norm(box[0] - box[3]))
            if height >= _LONG * width and height >= _MIN_WIDTH:
                vertical += 1
            elif width >= _LONG * height and width >= _MIN_WIDTH:
                horizontal += 1
    except Exception:  # noqa: BLE001 - es una optimizacion: nunca debe tumbar la lectura
        log.warning("No se pudo estimar si la tarjeta esta de lado", exc_info=True)
        return False
    return vertical >= MIN_LINES and vertical > horizontal
