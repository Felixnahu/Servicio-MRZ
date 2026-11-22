"""OCR de campos impresos: fechas y extraccion por etiqueta, con cajas sinteticas (sin OCR real)."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from servicio_mrz import visual
from servicio_mrz.app import create_app
from servicio_mrz.config import Settings
from servicio_mrz.ocr import MrzReader, Reading
from servicio_mrz.visual import Box

TODAY = date(2026, 11, 20)
W, H = 1024.0, 646.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("15/03/1990", date(1990, 3, 15)),
        ("15 / 03 / 1990", date(1990, 3, 15)),
        ("02/MAYO/MAY/1970", date(1970, 5, 2)),  # pasaporte: mes en espanol e ingles
        ("26/NOV/NOV/2015", date(2015, 11, 26)),
        ("Nacido el 5 de Marzo de 1985", date(1985, 3, 5)),  # cedula antigua
        ("O2/O5/l970", date(1970, 5, 2)),  # el OCR confunde O/0 y l/1 en las cifras
    ],
)
def test_parse_date_formats(text, expected):
    assert visual.parse_date(text, TODAY) == expected


@pytest.mark.parametrize(
    "text", ["31/02/1990", "29/08/2031", "29/13/1994", "sin fecha", "12/12/1850"]
)
def test_parse_date_rejects_impossible_or_implausible(text):
    assert visual.parse_date(text, TODAY) is None


def _ci_front() -> list[Box]:
    def b(x, y, text, w=0.25, h=0.04):
        return Box(x * W, y * H, (x + w) * W, (y + h) * H, text)

    return [
        b(0.15, 0.04, "ESTADO PLURINACIONAL DE BOLIVIA", 0.5),
        b(0.78, 0.07, "CEDULA DE IDENTIDAD", 0.2),
        b(0.74, 0.14, "N° 1234567", 0.2),
        b(0.55, 0.30, "NOMBRES:", 0.1),
        b(0.55, 0.35, "MARIA LUISA", 0.2),
        b(0.55, 0.43, "APELLIDOS:", 0.1),
        b(0.55, 0.48, "TORRES QUISPE", 0.3),
        b(0.55, 0.57, "FECHA DE NACIMIENTO:", 0.25),
        b(0.55, 0.62, "15/03/1990", 0.15),
        b(0.55, 0.75, "FECHA DE EMISION:", 0.2),
        b(0.55, 0.80, "16/12/2025", 0.15),
    ]


def test_ci_2023_front_fields_are_read_by_label():
    res = visual.parse_fields(_ci_front(), W, H)
    assert res is not None
    f = res["fields"]
    assert (f["firstNames"], f["lastNames"]) == ("MARIA LUISA", "TORRES QUISPE")
    assert f["birthDate"] == "1990-03-15" and f["documentNumber"] == "1234567"
    assert res["documentType"] == "CI_BO" and f["nationality"] == "BOL"
    # no tiene digitos de control: debe ir marcado para revision
    assert res["source"] == "ocr" and res["status"] == "REVIEW"
    assert "SIN_DIGITOS_DE_CONTROL" in res["reasons"]


def test_upside_down_card_gives_the_same_fields():
    flipped = [Box(W - b.x1, H - b.y1, W - b.x0, H - b.y0, b.text) for b in _ci_front()]
    res = visual.parse_fields(flipped, W, H)
    assert res is not None and res["fields"]["birthDate"] == "1990-03-15"


def test_the_date_below_the_wrong_label_is_not_taken():
    """Sin la etiqueta de nacimiento no se debe tomar la fecha de emision como nacimiento."""
    boxes = [b for b in _ci_front() if "NACIMIENTO" not in b.key and b.text != "15/03/1990"]
    assert visual.parse_fields(boxes, W, H) is None


def test_old_ci_back_reads_name_and_birth_date_written_in_words():
    def b(x, y, text, w=0.3, h=0.04):
        return Box(x * W, y * H, (x + w) * W, (y + h) * H, text)

    boxes = [
        b(0.17, 0.08, "CERTIFICA: Que la firma, fotografia", 0.54),
        b(0.03, 0.29, "A:", 0.04),
        b(0.28, 0.29, "LUCIA FERNANDA ROJAS", 0.4),
        b(0.04, 0.40, "Nacido el 5 de Marzo de 1985", 0.42),
    ]
    res = visual.parse_fields(boxes, W, H)
    assert res is not None and res["fields"]["birthDate"] == "1985-03-05"
    # el reverso trae el nombre completo: la division es una convencion (2 ultimos = apellidos)
    assert res["fields"]["firstNames"] == "LUCIA"
    assert res["fields"]["lastNames"] == "FERNANDA ROJAS"
    assert "DIVISION_DE_NOMBRES_ESTIMADA" in res["reasons"] and "NUMERO_NO_LEIDO" in res["reasons"]


def test_spaces_are_rebuilt_from_character_gaps():
    chars = list("AB CD")
    # 'AB' y 'CD' separadas por un hueco grande aunque el texto reconocido no traiga espacio
    boxes = [
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        [[11, 0], [21, 0], [21, 10], [11, 10]],
        [[22, 0], [32, 0], [32, 10], [22, 10]],
        [[50, 0], [60, 0], [60, 10], [50, 10]],
        [[61, 0], [71, 0], [71, 10], [61, 10]],
    ]
    assert visual._with_spaces("ABCD", boxes, list("ABxCD")) != "ABCD"
    assert visual._with_spaces("AB CD", boxes, chars) == "AB CD"


def test_endpoint_falls_back_to_printed_ocr_when_there_is_no_mrz(monkeypatch):
    printed = visual.parse_fields(_ci_front(), W, H)
    monkeypatch.setattr(MrzReader, "read_detailed", lambda self, img, *a, **k: Reading(None))
    monkeypatch.setattr(MrzReader, "read_visual", lambda self, img: printed)
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.full((400, 600, 3), 200, np.uint8))
    client = TestClient(create_app(Settings(camera_enabled=False, _env_file=None)))
    with client as c:
        res = c.post("/mrz/leer", files={"file": ("f.jpg", buf.tobytes(), "image/jpeg")})
    assert res.status_code == 200
    body = res.json()
    assert body["source"] == "ocr" and body["status"] == "REVIEW"
    assert body["fields"]["birthDate"] == "1990-03-15" and "kycMs" in body


def test_endpoint_is_422_when_neither_mrz_nor_printed_text_is_read(monkeypatch):
    monkeypatch.setattr(MrzReader, "read_detailed", lambda self, img, *a, **k: Reading(None))
    monkeypatch.setattr(MrzReader, "read_visual", lambda self, img: None)
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.full((400, 600, 3), 200, np.uint8))
    client = TestClient(create_app(Settings(camera_enabled=False, _env_file=None)))
    with client as c:
        res = c.post("/mrz/leer", files={"file": ("f.jpg", buf.tobytes(), "image/jpeg")})
    assert res.status_code == 422 and res.json()["codigo"] == "DOCUMENTO_NO_LEGIBLE"


def test_a_small_word_gap_is_only_seen_with_the_loose_threshold():
    """Palabras pegadas: el hueco (4 px, 0.4 del ancho de letra) no pasa el umbral normal (0.6)
    pero si el suelto (0.3), que solo se usa como respaldo (Box.loose)."""
    chars = list("ABCDEF")
    boxes = []
    x = 0
    for i in range(6):
        boxes.append([[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]])
        x += 10 + (4 if i == 2 else 0)  # hueco de 4 px entre C y D
    assert visual._with_spaces("ABCDEF", boxes, chars) == "ABCDEF"
    assert visual._with_spaces("ABCDEF", boxes, chars, visual.LOOSE_GAP) == "ABC DEF"


def test_read_boxes_computes_the_loose_text_only_for_long_glued_words():
    class Engine:
        def __call__(self, img, return_word_box=False, use_cls=False):
            def chars_of(text):
                return [[[i * 10, 0], [i * 10 + 10, 0], [i * 10 + 10, 10], [i * 10, 10]]
                        for i in range(len(text))], list(text)

            long_text, short_text = "LUISANGELANDRADERAMOS", "ESTUDIANTE"
            out = []
            for k, text in enumerate((long_text, short_text)):
                cb, ch = chars_of(text)
                pts = [[0, k * 20], [len(text) * 10, k * 20], [len(text) * 10, k * 20 + 10],
                       [0, k * 20 + 10]]
                out.append([pts, text, 0.9, cb, ch])
            return out, None

    import numpy as np

    boxes, _, _ = visual.read_boxes(Engine(), np.zeros((100, 300, 3), np.uint8), use_cls=False)
    assert boxes[0].loose != "" and boxes[1].loose == ""  # la corta no se toca
