"""Prueba de humo del motor OCR con una MRZ sintetica (fuente Consolas, NO es OCR-B).

Solo comprueba que la cadena imagen -> OCR -> parser funciona. El acierto real sobre documentos
fisicos se mide con fotos reales en samples/ (ver tests/measure_samples.py).
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from servicio_mrz.ocr import MrzReader, clean_row

FONT = Path("C:/Windows/Fonts/consola.ttf")
TD3 = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C<3UTO6908061F9406236ZE184226B<<<<<14",
]


def _render(lines: list[str]) -> np.ndarray:
    font = ImageFont.truetype(str(FONT), 44)
    img = Image.new("RGB", (1500, 60 + 70 * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((30, 30 + 70 * i), line, font=font, fill="black")
    return np.array(img)[:, :, ::-1].copy()  # RGB -> BGR


def test_clean_row_turns_ocr_filler_into_chevrons():
    assert clean_row("P<UTOERIKSSON<<ANNAKKKKKKKK") == "P<UTOERIKSSON<<ANNA" + "<" * 8
    assert clean_row("ab c-1") == "ABC<1"  # espacios fuera, simbolos raros -> '<'


@pytest.mark.skipif(not FONT.exists(), reason="requiere la fuente Consolas de Windows")
def test_synthetic_td3_is_read_end_to_end():
    result = MrzReader().read(_render(TD3))
    assert result is not None
    assert result.core_checks_ok, result.checks
    assert result.document_number == "L898902C"
