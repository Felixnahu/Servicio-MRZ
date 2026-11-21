"""Lectura por PLANTILLA de la cedula boliviana antigua (la de 1975 y la de 2020), sin MRZ.

Las dos versiones comparten formato; la unica diferencia es que la de 1975 trae un codigo de
barras y la de 2020 un QR en el anverso. Los datos estan repartidos entre las dos caras:

- REVERSO: "A: <nombre completo>" (una sola linea) y "Nacido el 15 de Marzo de 1985".
- ANVERSO: "No. 1234567" (numero de carnet, en rojo claro, abajo a la izquierda).

Estrategia (medida en 12 fotos de las dos versiones, ver tests/measure_old_ci.py):

1. Se recorta la zona del numero del anverso y se lee solo eso (rapido, ~0.5 s en un PC rapido).
2. El reverso se lee entero: recortar una banda con el nombre y la fecha daba peores resultados
   que la tarjeta completa (la fecha fallaba en 2 de 5). Se parsea por patron de fecha, no por
   etiqueta: el OCR deforma "Nacido" y "CERTIFICA" a menudo.

Igual que el OCR de campos impresos, NO hay digitos de control: el resultado va marcado `REVIEW` y
solo se devuelve si se leyo una fecha de nacimiento real. El numero se lee en ~60 % de los
anversos (el rojo claro sobre beige a veces ni se detecta): si falta, se avisa y se completa a mano.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import cv2
import numpy as np

from servicio_mrz import visual
from servicio_mrz.visual import Box

# Zona del anverso con "No. 1234567": ((y0, y1), (x0, x1)) como fraccion de la tarjeta enderezada.
FRONT_REGION = ((0.55, 0.80), (0.0, 0.6))
_NOT_NAME = ("PERTENECE", "IMPRESION", "CERTIFICA", "FIRMA", "SERVICIO", "IDENTIFICACION")
_MIN_YEAR = 1900


@dataclass
class OldCiRead:
    born: date | None = None
    full_name: str = ""
    number: str = ""

    def merge(self, other: OldCiRead) -> None:
        self.born = self.born or other.born
        self.full_name = self.full_name or other.full_name
        self.number = self.number or other.number


def _strip_label(text: str) -> str:
    """Quita un 'A:' inicial si el OCR lo pego al nombre ('A: PAULA ...')."""
    words = text.split()
    if words and re.fullmatch(r"[AÁ][:.;]?", words[0].upper()):
        words = words[1:]
    return " ".join(words)


def parse_back(boxes: list[Box], width: float, today: date | None = None) -> OldCiRead | None:
    """Nombre y fecha de nacimiento del reverso a partir de las cajas de la tarjeta."""
    dated = []
    for b in boxes:
        born = visual.parse_date(b.text, today)
        if born is not None and born.year >= _MIN_YEAR:
            dated.append((b, born))
    if not dated:
        return None
    line, born = min(dated, key=lambda item: item[0].cy)  # la fecha de nacimiento es la 1.a
    candidates = []
    for b in boxes:
        if b.cy >= line.cy - 0.3 * line.h or b.x0 < 0.12 * width:
            continue  # debajo de la fecha, o la etiqueta 'A:' del margen
        text = _strip_label(visual._name(b.text))
        key = visual._key(b.text)
        if any(w in key for w in _NOT_NAME) or any(c.isdigit() for c in b.text):
            continue
        if len(text.split()) >= 2 and len(text) >= 6:
            candidates.append((b, text))
    name = max(candidates, key=lambda item: item[0].cy)[1] if candidates else ""
    return OldCiRead(born=born, full_name=name)


def parse_front_number(boxes: list[Box]) -> str:
    """Numero de carnet: la primera cifra de 5 a 9 digitos de la zona 'No.'."""
    for b in sorted(boxes, key=lambda b: (b.y0, b.x0)):
        fixed = b.text.upper().translate(visual._DIGIT_FIX)
        for m in re.finditer(r"\d[\d ]{3,10}\d", fixed):
            digits = re.sub(r"\D", "", m[0])
            if 5 <= len(digits) <= 9:
                return digits
    return ""


def _front_number(engine: Any, card: np.ndarray) -> str:
    """Lee solo la zona del numero. El rojo claro es oscuro en el canal verde."""
    h, w = card.shape[:2]
    (y0, y1), (x0, x1) = FRONT_REGION
    green = card[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w), 1]
    green = cv2.resize(green, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    boxes, _, _ = visual.read_boxes(
        engine, cv2.cvtColor(green, cv2.COLOR_GRAY2BGR), use_cls=False
    )
    return parse_front_number(boxes)


def _back(engine: Any, card: np.ndarray) -> OldCiRead | None:
    boxes, width, _ = visual.read_boxes(engine, card, use_cls=False)
    return parse_back(boxes, width)


def read_old_ci(engine: Any, images: list[np.ndarray]) -> dict[str, Any] | None:
    """Datos de una cedula antigua a partir de sus fotos (frente y/o reverso, en cualquier orden).

    None si no se logra la fecha de nacimiento: el llamador pasa al respaldo.
    """
    cards = [visual._prepare(img) for img in images]
    total = OldCiRead()
    # 1.a pasada (barata): el numero del anverso. Una cara con numero legible es el anverso.
    front_idx: set[int] = set()
    for i, card in enumerate(cards):
        number = _front_number(engine, card)
        if number:
            total.number = total.number or number
            front_idx.add(i)
    # 2.a pasada: la fecha y el nombre, del reverso. Se prueba primero la ultima foto (la app pide
    # el reverso en 2.o lugar); la tarjeta puede estar boca abajo.
    for i in reversed(range(len(cards))):
        if total.born is not None:
            break
        if i in front_idx:
            continue
        part = _back(engine, cards[i]) or _back(engine, cv2.rotate(cards[i], cv2.ROTATE_180))
        if part is not None:
            total.merge(part)
    if total.born is None:
        return None
    words = total.full_name.split()
    cut = max(len(words) - 2, 1) if len(words) > 1 else len(words)  # 2 ultimos = apellidos
    return visual._result(
        "CI_BO",
        " ".join(words[:cut]),
        " ".join(words[cut:]),
        total.number,
        "BOL",
        total.born,
        "ci-old",
        extra_reasons=("DIVISION_DE_NOMBRES_ESTIMADA",) if len(words) > 1 else (),
    )
