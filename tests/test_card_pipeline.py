"""Camino por tarjeta con escenas sinteticas (fuente Consolas, NO OCR-B, imagen limpia).

Comprueban la logica (localizar, enderezar, variantes, celdas, controles), no la calidad real
frente a fotos de documentos fisicos; eso lo mide tests/measure_samples.py.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from servicio_mrz import preprocess
from servicio_mrz.ocr import MrzReader

FONT = Path("C:/Windows/Fonts/consola.ttf")
TD3 = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C<3UTO6908061F9406236ZE184226B<<<<<14",
]
pytestmark = pytest.mark.skipif(not FONT.exists(), reason="requiere la fuente Consolas de Windows")


def _passport_scene(angle: float, scale: float) -> np.ndarray:
    """Pagina de pasaporte (125x88 mm) con MRZ TD3 abajo, girada, sobre fondo oscuro."""
    w = 1000
    h = int(w * 88 / 125)
    page = Image.new("RGB", (w, h), (235, 232, 225))
    draw = ImageDraw.Draw(page)
    draw.rectangle([40, 60, 300, 300], fill=(190, 190, 200))
    # proporciones ICAO reales: paso de 20.2 px a 1024 px de ancho. Cada caracter se coloca con ese
    # paso exacto (el avance entero de pixel de Consolas lo desviaria ~1.5 % y acumularia deriva).
    pitch = 0.0197 * w
    font = ImageFont.truetype(str(FONT), round(pitch / 0.55))  # Consolas: paso = 0.55 em
    for i, line in enumerate(TD3):
        y = int(h * 0.78) + i * int(h * 0.1)
        for k, ch in enumerate(line):
            draw.text((w * 0.065 + k * pitch, y), ch, fill="black", font=font)
    img = cv2.resize(np.array(page)[:, :, ::-1].copy(), None, fx=scale, fy=scale)
    canvas = np.full((900, 1400, 3), 40, np.uint8)
    ph, pw = img.shape[:2]
    canvas[100 : 100 + ph, 100 : 100 + pw] = img
    rot = cv2.getRotationMatrix2D((100 + pw / 2, 100 + ph / 2), angle, 1)
    out = cv2.warpAffine(canvas, rot, (1400, 900), borderValue=(40, 40, 40))
    return cv2.resize(out[:720, :1280], (1280, 720))


@pytest.mark.parametrize("angle", [0, 4, -6])
def test_synthetic_passport_is_read_through_card_path(angle):
    reading = MrzReader().read_detailed(_passport_scene(angle, 0.6))
    assert reading.card is not None  # se localizo como tarjeta (no cayo al OCR general)
    assert reading.checks_ok
    assert reading.result.document_number == "L898902C"


def test_variants_keep_shape_and_are_distinct():
    card = np.random.default_rng(0).integers(0, 255, (300, 1024, 3), dtype=np.uint8)
    outs = {name: fn(card) for name, fn in preprocess.VARIANTS.items()}
    assert list(outs) == ["enfoque", "original", "clahe", "iluminacion"]
    for out in outs.values():
        assert out.shape == card.shape and out.dtype == np.uint8
    assert not np.array_equal(outs["enfoque"], outs["clahe"])


def test_cascade_reports_which_variant_read(monkeypatch):
    from servicio_mrz import celldecode, mrzread

    calls: list[str] = []
    real = mrzread._read_variant

    def spy(*args):
        calls.append("x")
        return real(*args)

    monkeypatch.setattr(mrzread, "_read_variant", spy)
    reader = MrzReader()
    engine = reader._engine_ready()
    recognizer = celldecode.LineRecognizer(engine)
    from servicio_mrz import cardfinder

    scene = _passport_scene(0, 0.6)
    located = cardfinder.locate_card(scene)
    card = cardfinder.warp_card(scene, located).image
    read = mrzread.read_card(engine, recognizer, card)
    assert read.result and read.result.core_checks_ok
    assert read.method.split(":")[-1] in preprocess.VARIANTS
    assert len(calls) <= len(preprocess.VARIANTS)  # se detiene en la primera que valida


def test_dark_bottom_border_does_not_hide_mrz_lines():
    """Regresion: una fila de borde (tinta de lado a lado) hacia perder las lineas reales."""
    from servicio_mrz import mrzlines

    card = np.full((646, 1024, 3), 235, np.uint8)
    font = ImageFont.truetype(str(FONT), 40)
    pil = Image.fromarray(card)
    draw = ImageDraw.Draw(pil)
    for i, line in enumerate(["I<UTOD231458907<<<<<<<<<<<<<<<", "7408122F1204159UTO<<<<<<<<<<<6"]):
        draw.text((110, 480 + i * 55), line, fill="black", font=font)
    card = np.array(pil)
    card[-10:] = 20  # borde oscuro de lado a lado, como tras enderezar con la esquina desplazada
    assert len(mrzlines.segment_lines(card)) == 2
    assert len(mrzlines.segment_lines(card, drop_border_rows=False)) < 2  # sin el arreglo fallaba


def test_incoherent_quad_is_rejected():
    """Regresion: lados opuestos de 1010 y 700 px (esquina mal propuesta) no es una tarjeta."""
    from servicio_mrz import cardfinder

    quad = np.array([[41, 17], [1548, 31], [1600, 731], [14, 1027]], np.float32)
    sides = [
        cardfinder._Side(
            quad[i],
            (quad[(i + 1) % 4] - quad[i]) / np.linalg.norm(quad[(i + 1) % 4] - quad[i]),
            1.0,
        )
        for i in range(4)
    ]
    assert cardfinder._verify(quad, sides, 1600.0 * 1033, 0.04) is None


def _scores(texts: list[str], noise: dict[tuple[int, int], str] | None = None) -> list[np.ndarray]:
    """Probabilidades por celda con la mayoria en el caracter dado; `noise` mete un error de OCR."""
    from servicio_mrz.celldecode import ALPHABET

    out = []
    for i, t in enumerate(texts):
        m = np.full((len(t), len(ALPHABET)), 0.01, np.float64)
        for j, ch in enumerate(t):
            shown = (noise or {}).get((i, j), ch)
            m[j, ALPHABET.index(shown)] = 0.6
            if shown != ch:  # el correcto queda como segunda opcion
                m[j, ALPHABET.index(ch)] = 0.35
        out.append(m)
    return out


def test_snap_fixes_document_type_and_country_from_the_format():
    """Regresion: un pasaporte con tipo 'AA' salia como CI_BO y 'BGR' como FOREIGN_ID."""
    from servicio_mrz import celldecode

    truth = ["P<BOLTORRES<QUISPE<<MARIA<LUISA<<<<<<<<<<<<<<", "X" * 10 + "BOL" + "<" * 31]
    truth = [truth[0][:44], truth[1][:44]]
    noisy = {(0, 0): "A", (0, 1): "A", (0, 3): "G", (0, 4): "R"}
    texts = [t for t in truth]
    for (line, pos), ch in noisy.items():
        texts[line] = texts[line][:pos] + ch + texts[line][pos + 1 :]
    fixed = celldecode.snap_fields(texts, _scores(truth, noisy), 2)
    assert fixed[0][:5] == "P<BOL"  # TD3 -> tipo 'P', emisor igual a la nacionalidad


def test_snap_does_not_turn_a_td1_into_a_passport():
    from servicio_mrz import celldecode

    truth = ["I<BOL" + "<" * 25, "<" * 15 + "BOL" + "<" * 12, "A" * 30]
    fixed = celldecode.snap_fields(truth, _scores(truth, {(0, 0): "P"}), 3)
    assert fixed[0][0] in "IAC"


def test_a_thin_bottom_strip_is_the_card_edge_not_an_mrz_line():
    """Reverso de cedula con 3 lineas MRZ + el borde inferior de la tarjeta: las 3 lineas MRZ son
    las 3 primeras, no las 3 ultimas (sin esto, la cedula del totem no se leia)."""
    from servicio_mrz import mrzread
    from servicio_mrz.mrzlines import LineSegment

    segs = [
        LineSegment(524, 552, 123, 986),
        LineSegment(575, 607, 73, 986),
        LineSegment(629, 662, 43, 981),
        LineSegment(690, 707, 40, 1005),  # 17 px frente a ~30: borde de la tarjeta
    ]
    first_three = mrzread.hypotheses(segs, 707)[0]
    assert first_three[1] == 3 and [s.y0 for s in first_three[0]] == [524, 575, 629]
    # una banda final de grosor normal SI cuenta como linea
    segs[-1] = LineSegment(690, 720, 40, 1005)
    assert [s.y0 for s in mrzread.hypotheses(segs, 740)[0][0]] == [575, 629, 690]
