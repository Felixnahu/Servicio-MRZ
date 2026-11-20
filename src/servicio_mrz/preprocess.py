"""Variantes de preprocesado de la tarjeta ya enderezada, para probarlas en cascada.

Ninguna es "la mejor": segun luz, reflejo y desenfoque gana una u otra. Como los digitos de control
validan la lectura, se puede probar cada variante y quedarse con la primera que valide; un error
de preprocesado nunca produce un dato falso, solo una lectura mas lenta.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


def _gray3(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def original(card: np.ndarray) -> np.ndarray:
    return card


def gray_clahe(card: np.ndarray) -> np.ndarray:
    """Grises + ecualizacion adaptativa (CLAHE, 'clip limit'): recupera contraste local."""
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    return _gray3(cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 8)).apply(gray))


def sharpen(card: np.ndarray) -> np.ndarray:
    """Grises + mascara de enfoque: ayuda con desenfoque leve."""
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    return _gray3(cv2.addWeighted(gray, 1.8, blur, -0.8, 0))


def flatfield(card: np.ndarray) -> np.ndarray:
    """Corrige iluminacion desigual: divide por el fondo estimado (cierre morfologico).

    Recupera el texto tenue bajo reflejos suaves o degradados (medido: +2 de 15 cedulas).
    """
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    k = 31
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    norm = np.clip(gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1) * 255, 0, 255)
    lo, hi = np.percentile(norm, [1, 99])
    return _gray3((np.clip((norm - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8))


VARIANTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # orden medido (tests/measure_variants.py): la mas rapida y util primero
    "enfoque": sharpen,
    "original": original,
    "clahe": gray_clahe,
    "iluminacion": flatfield,
}
