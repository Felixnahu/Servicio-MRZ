"""Lectura por plantilla de la cedula antigua: extraccion con cajas sinteticas (sin OCR real)."""

from datetime import date

import numpy as np
import pytest
from fastapi.testclient import TestClient

from servicio_mrz import oldci
from servicio_mrz.app import create_app
from servicio_mrz.config import Settings
from servicio_mrz.ocr import MrzReader, Reading
from servicio_mrz.visual import Box

W, H = 1024.0, 690.0


def _b(x, y, text, w=0.3, h=0.045):
    return Box(x * W, y * H, (x + w) * W, (y + h) * H, text)


def _back_boxes(name="MARIA LUISA TORRES QUISPE", born="Nacido el 15 de Marzo de 1985"):
    return [
        _b(0.05, 0.02, "EL SERVICIO GENERAL DE IDENTIFICACION PERSONAL", 0.8),
        _b(0.18, 0.08, "CERTIFICA: Que la firma, fotografia", 0.65),
        _b(0.29, 0.15, "e impresion pertenece", 0.4),
        _b(0.02, 0.26, "A:", 0.04),
        _b(0.29, 0.26, name, 0.48),
        _b(0.02, 0.38, born, 0.5),
        _b(0.03, 0.44, "En", 0.04),
        _b(0.32, 0.44, "SANTA CRUZ - ANDRES IBANEZ", 0.4),
        _b(0.03, 0.50, "Estado Civil SOLTERA", 0.35),
    ]


def test_back_gives_birth_date_and_the_full_name_line():
    got = oldci.parse_back(_back_boxes(), W, today=date(2026, 11, 21))
    assert got is not None
    assert got.born == date(1985, 3, 15) and got.full_name == "MARIA LUISA TORRES QUISPE"


def test_back_still_reads_when_ocr_mangles_the_labels():
    """El OCR deforma 'Nacido' y 'CERTIFICA' a menudo: se busca por patron de fecha."""
    boxes = _back_boxes(born="Nacldo el 15 de Marzo de 1985")
    boxes[1] = _b(0.18, 0.08, "CERTlF1CA: Que la firma", 0.6)
    boxes[3] = _b(0.02, 0.26, "xx", 0.04)  # la etiqueta 'A:' no se leyo
    got = oldci.parse_back(boxes, W, today=date(2026, 11, 21))
    assert got is not None and got.born == date(1985, 3, 15)
    assert got.full_name == "MARIA LUISA TORRES QUISPE"


def test_back_reads_month_when_final_o_is_seen_as_zero():
    got = oldci.parse_back(
        _back_boxes(born="Nacido el 31de Marz0 de 2003"), W, today=date(2026, 11, 21)
    )
    assert got is not None and got.born == date(2003, 3, 31)


def test_a_name_glued_to_the_label_loses_the_label():
    got = oldci.parse_back(_back_boxes(name="A: MARIA LUISA TORRES QUISPE"), W, date(2026, 11, 21))
    assert got is not None and got.full_name == "MARIA LUISA TORRES QUISPE"


def test_back_without_a_valid_date_is_not_a_read():
    assert oldci.parse_back(_back_boxes(born="Nacido el 31 de Febrero de 1985"), W) is None
    assert oldci.parse_back([_b(0.1, 0.3, "DOMICILIO AV. CENTENARIO")], W) is None


def test_back_rejects_passport_date_without_old_ci_text():
    boxes = [
        _b(0.1, 0.03, "ESTADO PLURINACIONAL DE BOLIVIA"),
        _b(0.2, 0.25, "JUAN PABLO SALINAS SALEK"),
        _b(0.2, 0.38, "08/DIC/1988"),
    ]
    assert oldci.parse_back(boxes, W) is None


def test_back_rejects_new_ci_front_even_with_servicio_general_heading():
    boxes = [
        _b(0.1, 0.03, "SERVICIO GENERAL DE IDENTIFICACION PERSONAL"),
        _b(0.4, 0.30, "NOMBRES: ALVARO"),
        _b(0.4, 0.40, "FECHA DE NACIMIENTO: 26/02/1987"),
    ]
    assert oldci.parse_back(boxes, W) is None


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (["No.", "1234567"], "1234567"),
        (["No. 654321"], "654321"),
        (["No.", "12S4567"], "1254567"),  # S leida por 5
        (["serie", "11111"], "11111"),  # 5 digitos tambien valen (cedulas antiguas cortas)
        (["No.", "290"], ""),  # demasiado corto
        (["Emitida el", "3 de Enero"], ""),
    ],
)
def test_front_number(texts, expected):
    boxes = [_b(0.05 + 0.2 * i, 0.66, t, 0.15) for i, t in enumerate(texts)]
    assert oldci.parse_front_number(boxes) == expected


def test_front_number_retries_color_only_when_number_label_is_seen(monkeypatch):
    calls = []

    def fake_boxes(engine, image, use_cls=False):
        calls.append(image)
        boxes = [_b(0.05, 0.66, "No.")] if len(calls) == 1 else [_b(0.05, 0.66, "1234567")]
        return boxes, image.shape[1], image.shape[0]

    monkeypatch.setattr(oldci.visual, "read_boxes", fake_boxes)
    card = np.full((400, 600, 3), 200, np.uint8)
    assert oldci._front_number(object(), card) == "1234567"
    assert len(calls) == 2

    calls.clear()
    monkeypatch.setattr(oldci.visual, "read_boxes", lambda *args, **kwargs: ([], 600, 400))
    assert oldci._front_number(object(), card) == ""


def test_endpoint_uses_the_local_old_ci_reader(monkeypatch):
    import cv2

    calls = []

    def fake_old(self, images, deadline=None, detect_student=True):
        calls.append(len(images))
        return {"documentType": "CI_BO", "fields": {"birthDate": "1985-03-15"}, "source": "ocr"}

    corners = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))
    aspects = iter((1.44, 1.46))
    monkeypatch.setattr(
        MrzReader, "read_detailed",
        lambda self, img, *a, **k: Reading(None, card=corners, card_aspect=next(aspects)),
    )
    monkeypatch.setattr(MrzReader, "read_old_ci", fake_old)
    ok, buf = cv2.imencode(".jpg", np.full((400, 600, 3), 200, np.uint8))
    files = {"front": ("f.jpg", buf.tobytes()), "back": ("b.jpg", buf.tobytes())}
    settings = Settings(camera_enabled=False, visual_ocr=False, _env_file=None)
    with TestClient(create_app(settings)) as c:
        res = c.post("/mrz/leer", data={"documentType": "CI_BO"}, files=files)
    assert res.status_code == 200 and res.json()["source"] == "ocr" and calls == [2]
    assert "duracionMs" in res.json()


def test_old_ci_reader_is_tried_for_ambiguous_shape_and_false_mrz_observation(monkeypatch):
    import cv2

    from servicio_mrz.mrzread import Observation

    calls = []
    monkeypatch.setattr(MrzReader, "read_old_ci", lambda self, imgs, *a, **k: calls.append(1))
    ok, buf = cv2.imencode(".jpg", np.full((400, 600, 3), 200, np.uint8))
    files = {"front": ("f.jpg", buf.tobytes())}
    settings = Settings(camera_enabled=False, visual_ocr=False, _env_file=None)

    corners = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))
    ambiguous = Reading(None, card=corners, card_aspect=1.40)
    monkeypatch.setattr(MrzReader, "read_detailed", lambda self, img, *a, **k: ambiguous)
    with TestClient(create_app(settings)) as c:
        c.post("/mrz/leer", files=files)  # proporciones de cedula antigua pueden ser 1.44
    assert calls == [1]

    calls.clear()
    seen = Reading(None, observation=Observation(2, (np.zeros((44, 3)),), 0.9))
    monkeypatch.setattr(MrzReader, "read_detailed", lambda self, img, *a, **k: seen)
    with TestClient(create_app(settings)) as c:
        c.post("/mrz/leer", data={"documentType": "CI_BO"}, files=files)
    assert calls == [1]


def test_old_ci_reader_stops_at_the_deadline():
    """Con el plazo vencido no se hace ningun OCR (en el totem pasaba de 40 s y la app abandona)."""
    import time

    class Boom:
        def __call__(self, *a, **k):
            raise AssertionError("no debe leer nada tras el plazo")

    img = np.full((400, 600, 3), 200, np.uint8)
    assert oldci.read_old_ci(Boom(), [img, img], deadline=time.monotonic() - 1) is None


def test_old_ci_reader_reads_required_back_before_optional_front(monkeypatch):
    calls = []

    monkeypatch.setattr(oldci.visual, "_prepare", lambda img: img)

    def fake_back(engine, card):
        calls.append("back")
        return oldci.OldCiRead(born=date(1985, 3, 15), full_name="MARIA LUISA TORRES QUISPE")

    monkeypatch.setattr(oldci, "_back", fake_back)
    monkeypatch.setattr(
        oldci, "_front_number", lambda engine, card: calls.append("front") or "1234567"
    )
    img = np.full((400, 600, 3), 200, np.uint8)
    result = oldci.read_old_ci(object(), [img, img])
    assert result is not None and result["fields"]["birthDate"] == "1985-03-15"
    assert result["fields"]["documentNumber"] == "1234567"
    assert calls == ["back", "front"]


def test_upside_down_card_is_straightened_before_the_first_ocr_pass(monkeypatch):
    """Si el clasificador de giro dice que esta al reves, la 1.a pasada ya lee la tarjeta girada."""
    seen = []
    card = np.zeros((40, 60, 3), np.uint8)
    card[:5] = 255  # marca arriba: tras girar 180 grados queda abajo
    monkeypatch.setattr(oldci.visual, "_prepare", lambda img: card)
    # siempre "al reves", tambien tras girar la foto: se cae al giro de la tarjeta enderezada
    monkeypatch.setattr(oldci.orientation, "upside_down", lambda engine, c: True)

    def fake_boxes(engine, c, use_cls=False):
        seen.append(bool(c[-1].any()))  # True = la marca quedo abajo = se giro
        return [], 60, 40

    monkeypatch.setattr(oldci.visual, "read_boxes", fake_boxes)
    assert oldci.read_old_ci(object(), [card]) is None
    assert seen[0] is True


def test_upright_card_relocates_the_flipped_photo_when_the_vote_confirms_it(monkeypatch):
    """Al reves: se gira la FOTO y se vuelve a localizar (mejor recorte que girar la tarjeta)."""
    upside = np.full((40, 60, 3), 10, np.uint8)
    straight = np.full((40, 60, 3), 20, np.uint8)
    prepared = {}

    def fake_prepare(img):
        out = straight if img[0, 0, 0] == 99 else upside  # 99 marca la foto ya girada
        prepared[id(out)] = img
        return out

    photo = np.zeros((40, 60, 3), np.uint8)
    flipped = photo.copy()
    flipped[:] = 99
    monkeypatch.setattr(oldci.visual, "_prepare", fake_prepare)
    monkeypatch.setattr(oldci.cv2, "rotate", lambda img, code: flipped)
    votes = iter([True, False])  # 1.a tarjeta al reves; la re-localizada, del derecho
    monkeypatch.setattr(oldci.orientation, "upside_down", lambda engine, c: next(votes))
    assert oldci._upright_card(object(), photo) is straight


def test_a_passport_page_is_never_an_old_ci_back():
    """'NACIONALIDAD' mal leida contiene 'NACID': no debe pasar por reverso de carnet antiguo."""
    boxes = [
        Box(10, 10, 300, 40, "NACIDNALIDAD / NATIONALITY"),
        Box(10, 50, 300, 80, "FECHA DE NACIMIENTO 08 DIC 1988"),
        Box(10, 90, 300, 120, "JUAN PEREZ GOMEZ"),
    ]
    assert oldci.parse_back(boxes, 600) is None


def test_back_number_is_taken_only_as_8_digits_at_the_left_margin_above_the_date():
    boxes = _back_boxes()
    boxes += [
        _b(0.03, 0.10, "11338643", 0.04),  # numero impreso arriba a la izquierda (8 digitos)
        _b(0.72, 0.10, "138239", 0.04),  # serie a la derecha: no cuenta
    ]
    got = oldci.parse_back(boxes, W, today=date(2026, 11, 21))
    assert got is not None and got.number == "11338643"


def test_a_six_digit_box_at_the_left_margin_is_not_a_document_number():
    boxes = _back_boxes() + [_b(0.03, 0.10, "138239", 0.04)]
    got = oldci.parse_back(boxes, W, today=date(2026, 11, 21))
    assert got is not None and got.number == ""


def test_a_number_below_the_date_line_is_ignored():
    boxes = _back_boxes() + [_b(0.03, 0.60, "11338643", 0.04)]
    got = oldci.parse_back(boxes, W, today=date(2026, 11, 21))
    assert got is not None and got.number == ""


def test_glued_name_is_recovered_with_the_loose_text():
    glued = _b(0.29, 0.26, "LUISANGELANDRADERAMOS", 0.48)
    boxes = [b for b in _back_boxes() if "MARIA" not in b.text] + [
        Box(glued.x0, glued.y0, glued.x1, glued.y1, glued.text, "LUIS ANGEL ANDRADE RAMOS")
    ]
    got = oldci.parse_back(boxes, W, today=date(2026, 11, 21))
    assert got is not None and got.full_name == "LUIS ANGEL ANDRADE RAMOS"


def test_glued_name_without_a_loose_text_is_still_not_guessed():
    boxes = [b for b in _back_boxes() if "MARIA" not in b.text] + [
        _b(0.29, 0.26, "LUISANGELANDRADERAMOS", 0.48)
    ]
    got = oldci.parse_back(boxes, W, today=date(2026, 11, 21))
    assert got is not None and got.full_name == ""
