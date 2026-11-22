"""Carnet universitario: lo detecta por las palabras que trae impresas (sin leer datos personales).

Medido con fotos reales (ver tests/measure_student.py): `ESTUDIANTE` aparece en ~1 de cada 3
cedulas y pasaportes porque es un valor del campo "profesion u ocupacion", y `CARRERA` esta a una
letra de `CABRERA`/`CARRERA`, apellidos comunes. Por eso NO bastan solas:

- Fuerte: UNIVERSITARIO o UNIVERSIDAD (tolera 1-2 letras mal leidas). Una sola basta.
- Fuerte: una palabra que EMPIEZA por UNIV (UNIVALLE, UNIVERSIDAD...), salvo UNIVERSAL/UNIVERSO,
  o un dominio `.edu` (cada universidad se escribe distinto: hay carnets sin "universitario").
- Debil: CARRERA, ESTUDIANTE y FACULTAD, EXACTAS. Hacen falta las tres a la vez (una cedula con
  apellido Carrera y ocupacion Estudiante tiene dos, pero no la facultad).
- Por estructura: hay carnets de estudiante (de instituto o de universidad que no se nombra) que
  traen solo ESTUDIANTE, el nombre y el CI en pocas lineas. Cuenta ESTUDIANTE cuando la tarjeta
  tiene pocas cajas de texto (<= MAX_BOXES_ALONE) y NINGUNA etiqueta de cedula o pasaporte
  (OCUPACION, PROFESION, DOMICILIO...): en una cedula la palabra es el valor de ese campo.

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
_NOT_UNIV = ("UNIVERSAL", "UNIVERSO")  # empiezan por UNIV y no son de universidad
_UNIV_WORD = re.compile(r"\bUNIV[A-Z]{2,}")
_EDU_DOMAIN = re.compile(r"\.\s?EDU\b")  # est.univalle.edu, upds.edu.bo
WEAK = ("CARRERA", "ESTUDIANTE", "FACULTAD")
# Etiquetas que solo traen una cedula o un pasaporte: con alguna, ESTUDIANTE es la ocupacion.
_ID_LABELS = (
    "OCUPACION", "PROFESION", "DOMICILIO", "ESTADOCIVIL", "NACIDO", "CERTIFICA", "IDENTIDAD",
    "CEDULA", "PASAPORTE", "PASSPORT", "SURNAME", "NATIONALITY", "SERVICIOGENERAL", "LUGARDE",
    # valores que solo salen en el reverso de una cedula, aunque el OCR no lea las etiquetas
    "SOLTERO", "CASADO", "VIUDO", "DIVORCIADO", "SANGUINEO",
)
_MRZ_FILLER = "<<"  # relleno de una MRZ ("I<BOL...<<<<"): ninguna tarjeta de estudiante lo trae
MAX_BOXES_ALONE = 12
STRONG_RATIO = 0.85  # 13 letras: tolera 1-2 errores del OCR


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
    raw = " ".join(b.text.upper() for b in boxes)
    if _EDU_DOMAIN.search(raw) or any(
        not w.startswith(_NOT_UNIV) for w in _UNIV_WORD.findall(raw)
    ):
        return True
    if all(w in key for w in WEAK):  # exactas: sin tolerancia (evita CABRERA)
        return True
    return (
        "ESTUDIANTE" in key
        and len(boxes) <= MAX_BOXES_ALONE
        and not any(label in key for label in _ID_LABELS)
        and _MRZ_FILLER not in raw
    )


def result() -> dict:
    """JSON del contrato: solo el TIPO (`STUDENT_CARD`); el resto de datos se completa a mano."""
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
        "suggestedCategory": None,  # el backend no tiene categoria 'universitario': va en el tipo
        "categoryMatches": None,
        "source": "ocr",
        "requiresManualEntry": False,  # un 200 nunca lo trae en true (ver contract.conform)
        "status": "REVIEW",
        "reasons": ["CARNET_UNIVERSITARIO_DETECTADO", "DATOS_NO_LEIDOS"],
    }
