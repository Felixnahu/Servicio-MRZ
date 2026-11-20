"""Limpieza del texto que devuelve el OCR para una linea MRZ."""

import re

_MRZ_CHARS = re.compile(r"[^A-Z0-9<]")
_TRAILING_FILLER = re.compile(r"[KC<]{3,}$")  # el OCR suele leer '<' repetidos como K o C


def clean_row(text: str) -> str:
    row = _MRZ_CHARS.sub("<", text.upper().replace(" ", ""))
    return _TRAILING_FILLER.sub(lambda m: "<" * len(m.group()), row)
