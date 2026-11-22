"""Carnet universitario: lo detecta por las palabras que trae impresas (sin leer datos personales).

Medido con fotos reales (ver tests/measure_student.py): `ESTUDIANTE` aparece en ~1 de cada 3
cedulas y pasaportes porque es un valor del campo "profesion u ocupacion", y `CARRERA` esta a una
letra de `CABRERA`/`CARRERA`, apellidos comunes. Por eso NO bastan solas:

- Fuerte: UNIVERSITARIO o UNIVERSIDAD (tolera 1-2 letras mal leidas). Una sola basta.
- Debil: CARRERA, ESTUDIANTE y FACULTAD, EXACTAS. Hacen falta las tres a la vez (una cedula con
  apellido Carrera y ocupacion Estudiante tiene dos, pero no la facultad).

Es una PISTA para que el backend valide la categoria, no una verificacion: cualquier papel con
esas palabras la dispara y no se comprueba la vigencia.
"""

from __future__ import annotations

import difflib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from servicio_mrz.visual import Box

STRONG = ("UNIVERSITARIO", "UNIVERSIDAD")
WEAK = ("CARRERA", "ESTUDIANTE", "FACULTAD")
STRONG_RATIO = 0.85  # 13 letras: tolera 1-2 errores del OCR
CATEGORY = "universitario"


def _letters(text: str) -> str:
    return re.sub(r"[^A-Z]", "", text.upper())


def _has(key: str, word: str) -> bool:
    if word in key:
        return True
    n = len(word)
    return any(
        difflib.SequenceMatcher(None, key[i : i + n], word).ratio() >= STRONG_RATIO
        for i in range(max(len(key) - n + 1, 1))
    )


def detect(boxes: list[Box]) -> bool:
    """True si las cajas de texto de una tarjeta son de un carnet universitario."""
    key = "".join(_letters(b.text) for b in boxes)
    if any(_has(key, w) for w in STRONG):
        return True
    return all(w in key for w in WEAK)  # exactas: sin tolerancia (evita CABRERA)


def result() -> dict:
    """JSON del contrato: solo la categoria; el resto de datos se completa a mano en la app."""
    return {
        "documentType": "STUDENT_CARD",
        "documentModel": "student-card",
        "fields": {
            "firstNames": "",
            "lastNames": "",
            "documentNumber": "",
            "nationality": "",
            "birthDate": "",
        },
        "age": None,
        "isAdult": None,
        "isSamaipateno": None,
        "suggestedCategory": CATEGORY,
        "categoryMatches": None,
        "source": "ocr",
        "requiresManualEntry": True,
        "status": "REVIEW",
        "reasons": ["CARNET_UNIVERSITARIO_DETECTADO", "DATOS_NO_LEIDOS"],
    }
