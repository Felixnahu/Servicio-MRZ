"""Lugar de nacimiento del carnet boliviano: decide si el titular nacio en Samaipata.

- Carnet ANTIGUO (1975/2020): el reverso dice "Nacido el 15 de Marzo de 1985 en" y el lugar va en
  la linea de abajo.
- Carnet NUEVO (2023): el reverso tiene la etiqueta "LUGAR DE NACIMIENTO" arriba y el valor justo
  debajo, en la misma columna.

La decision es deliberadamente asimetrica: `True` solo si el texto del lugar se parece a
"SAMAIPATA" (tolera errores del OCR), `False` solo si SE LEYO un lugar distinto, y `None` si no se
leyo nada. Un "no" por falta de lectura cobraria la tarifa equivocada a un vecino.
"""

from __future__ import annotations

import difflib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from servicio_mrz.visual import Box

TARGET = "SAMAIPATA"
MATCH_RATIO = 0.75  # 9 letras: tolera hasta 2 errores del OCR
MIN_LETTERS = 4  # menos letras que esto no es un lugar leido
BAND_BELOW = (0.4, 1.8)  # filas (en alturas de linea) bajo la etiqueta donde esta el valor
_LABEL_WORDS = ("NACIMIENTO", "FECHA", "LUGAR")


def _letters(text: str) -> str:
    return re.sub(r"[^A-Z]", "", text.upper())


def has_samaipata(text: str) -> bool:
    key = _letters(text)
    if TARGET in key:
        return True
    n = len(TARGET)
    return any(
        difflib.SequenceMatcher(None, key[i : i + n], TARGET).ratio() >= MATCH_RATIO
        for i in range(max(len(key) - n + 1, 0))
    )


def decide(text: str) -> bool | None:
    """True/False/None segun el texto del lugar de nacimiento (ver la regla arriba)."""
    if has_samaipata(text):
        return True
    return False if len(_letters(text)) >= MIN_LETTERS else None


def _below(boxes: list[Box], anchor: Box, min_x: float) -> list[Box]:
    lo, hi = BAND_BELOW
    rows = [
        b
        for b in boxes
        if b is not anchor
        and lo * anchor.h < b.cy - anchor.cy < hi * anchor.h
        and b.x0 >= min_x
        and not any(w in _letters(b.text) for w in _LABEL_WORDS)
    ]
    return sorted(rows, key=lambda b: (b.cy, b.x0))


def from_old_boxes(boxes: list[Box], date_line: Box) -> bool | None:
    """Carnet antiguo: la linea de la fecha ("...de 1985 en") y la de debajo."""
    after_year = re.search(r"(?:19|20)\d{2}(.*)$", date_line.text)  # "... de 1985 en <lugar?>"
    tail = after_year.group(1) if after_year else ""
    return decide(" ".join([tail, *(b.text for b in _below(boxes, date_line, 0.0))]))


def from_new_boxes(boxes: list[Box], width: float) -> bool | None:
    """Carnet 2023: el valor bajo la etiqueta "LUGAR DE NACIMIENTO". None si no hay etiqueta."""
    label = next((b for b in boxes if _is_label(b.text)), None)
    if label is None:
        return None
    values = _below(boxes, label, label.x0 - 0.03 * width)
    return decide(" ".join(b.text for b in values))


def _is_label(text: str) -> bool:
    key = _letters(text)
    if "LUGARDE" in key:
        return True
    n = 7
    return any(
        difflib.SequenceMatcher(None, key[i : i + n], "LUGARDE").ratio() >= 0.85
        for i in range(max(len(key) - n + 1, 0))
    )
