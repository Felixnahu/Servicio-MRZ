"""Detector de carnet universitario. Cajas sinteticas: sin datos de personas reales."""

import numpy as np

from servicio_mrz import oldci, student
from servicio_mrz.visual import Box


def boxes(*texts: str) -> list[Box]:
    return [Box(10, 10 + 30 * i, 300, 30 + 30 * i, t) for i, t in enumerate(texts)]


def test_one_strong_word_is_enough():
    assert student.detect(boxes("CARNET", "UNIVERSITARIO"))
    assert student.detect(boxes("UNIVERSIDAD AUTONOMA"))
    assert student.detect(boxes("UNIVERSITAR1O"))  # el OCR confunde letras: se tolera


def test_estudiante_alone_is_the_occupation_field_of_an_id_card():
    """En cedulas y pasaportes 'ESTUDIANTE' es el valor de PROFESION/OCUPACION (7 de 16 carnets)."""
    assert not student.detect(boxes("PROFESION U OCUPACION", "ESTUDIANTE"))


def test_a_surname_like_carrera_does_not_count():
    """CARRERA esta a una letra de CABRERA (apellido comun): no basta, ni con ESTUDIANTE."""
    assert not student.detect(boxes("APELLIDOS: CARRERA CABRERA"))
    # apellido + ocupacion, como sale en una cedula (con la etiqueta del campo)
    assert not student.detect(boxes("CARRERA", "PROFESION U OCUPACION", "ESTUDIANTE"))


def test_the_three_weak_words_together_do_count():
    assert student.detect(boxes("ESTUDIANTE:", "CARRERA:", "FACULTAD DE INGENIERIA"))


def test_result_has_the_contract_fields_and_asks_for_manual_entry():
    r = student.result()
    # el backend no tiene categoria universitaria (enum cerrado): la senal es el tipo de documento
    assert r["documentType"] == "STUDENT_CARD" and r["suggestedCategory"] is None
    assert r["requiresManualEntry"] is False and r["fields"]["birthDate"] == ""
    assert r["status"] == "REVIEW" and "DATOS_NO_LEIDOS" in r["reasons"]
    assert r["age"] is None and r["isSamaipateno"] is None and r["categoryMatches"] is None


def test_old_ci_reader_returns_the_student_result_without_extra_ocr(monkeypatch):
    monkeypatch.setattr(oldci.visual, "_prepare", lambda img: img)
    calls = []

    def fake_boxes(engine, card, use_cls=False):
        calls.append(1)
        return boxes("CARNET UNIVERSITARIO", "ESTUDIANTE:"), 600, 400

    monkeypatch.setattr(oldci.visual, "read_boxes", fake_boxes)
    img = np.full((400, 600, 3), 200, np.uint8)
    res = oldci.read_old_ci(object(), [img, img])
    assert res is not None and res["documentType"] == "STUDENT_CARD"
    assert len(calls) == 1  # se detiene en la primera lectura
    assert oldci.read_old_ci(object(), [img, img], detect_student=False) is None


def test_university_name_starting_with_univ_counts():
    """Hay carnets que no dicen 'universitario' (el de Univalle trae solo el nombre)."""
    assert student.detect(boxes("UNIVALLE", "SEDE ACADEMICA", "ING. BIOMEDICA"))
    assert student.detect(boxes("est.univalle.edu", "Sitio web"))


def test_edu_domain_counts_but_names_that_contain_edu_do_not():
    assert student.detect(boxes("InfoUpds.SantaCruz@upds.edu.bo"))
    assert not student.detect(boxes("NOMBRES: EDUARDO", "LUGAR: EDUCACION"))


def test_univ_words_that_are_not_about_universities_do_not_count():
    assert not student.detect(boxes("DECLARACION UNIVERSAL", "UNIVERSO"))


def test_both_photos_are_read_unrotated_before_any_rotated_pass(monkeypatch):
    """En el totem cada pasada de OCR tarda ~7 s y el plazo es corto: girar la 1.a foto antes de
    mirar la otra gastaba el plazo sin leer el anverso (caso real de un carnet universitario)."""
    monkeypatch.setattr(oldci.visual, "_prepare", lambda img: img)
    order = []

    def fake_back(engine, card):
        tag = int(card[0, 0, 0] or card[-1, -1, 0])
        order.append((tag, card[0, 0, 0] == 0))  # (que foto, girada?)
        return None

    monkeypatch.setattr(oldci, "_back", fake_back)
    a = np.zeros((40, 60, 3), np.uint8)
    a[0, 0, 0] = 1
    b = np.zeros((40, 60, 3), np.uint8)
    b[0, 0, 0] = 2
    assert oldci.read_old_ci(object(), [a, b]) is None
    assert order == [(2, False), (1, False), (2, True), (1, True)]


def test_with_time_for_two_passes_the_second_is_the_other_face(monkeypatch):
    """Con tiempo para 2 pasadas (como en el totem), la 2.a es la OTRA cara, no la misma girada."""
    import time

    monkeypatch.setattr(oldci.visual, "_prepare", lambda img: img)
    seen = []
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    def fake_back(engine, card):
        seen.append(int(card[0, 0, 0] or card[-1, -1, 0]))
        clock["t"] += 7.0  # lo que tarda una pasada en el totem
        return oldci.OldCiRead(student=True) if len(seen) == 2 else None

    monkeypatch.setattr(oldci, "_back", fake_back)
    a = np.zeros((40, 60, 3), np.uint8)
    a[0, 0, 0] = 1
    b = np.zeros((40, 60, 3), np.uint8)
    b[0, 0, 0] = 2
    res = oldci.read_old_ci(object(), [a, b], deadline=10.0)
    assert res is not None and res["documentType"] == "STUDENT_CARD"
    assert seen == [2, 1]  # reverso y despues anverso: la mitad del tiempo no se gasta girando


def test_estudiante_alone_counts_on_a_sparse_card_without_id_labels():
    """Carnet de estudiante sin 'universitario': pocas lineas, ESTUDIANTE, nombre y CI."""
    assert student.detect(boxes("NOMBRE APELLIDO", "ESTUDIANTE", "C.I. 1234567", "INSTITUTO"))


def test_estudiante_on_an_id_back_is_not_a_student_card_even_if_labels_are_unread():
    """Reverso de cedula: el OCR a veces no lee 'OCUPACION' pero si la MRZ o el estado civil."""
    assert not student.detect(boxes("ESTUDIANTE", "I<BOL1234567<<5<<<<<<<<<<<<<<<"))
    assert not student.detect(boxes("ESTUDIANTE", "SOLTERO", "O RH (+)"))
    assert not student.detect(boxes("ESTUDIANTE", "GRUPO SANGUINEO"))


def test_estudiante_on_a_crowded_page_is_not_a_student_card():
    texts = ["ESTUDIANTE"] + [f"LINEA {i}" for i in range(13)]
    assert not student.detect(boxes(*texts))


def test_endpoint_answers_a_student_card_in_the_shape_of_the_backend_dto(monkeypatch):
    """El 200 de un carnet universitario calza con `DocumentReadingResponseDto` del backend."""
    import cv2
    from fastapi.testclient import TestClient

    from servicio_mrz.app import create_app
    from servicio_mrz.config import Settings
    from servicio_mrz.ocr import MrzReader, Reading

    corners = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))
    monkeypatch.setattr(
        MrzReader, "read_detailed",
        lambda self, img, *a, **k: Reading(None, card=corners, card_aspect=1.5),
    )
    monkeypatch.setattr(MrzReader, "read_old_ci", lambda self, *a, **k: student.result())
    ok, buf = cv2.imencode(".jpg", np.full((400, 600, 3), 200, np.uint8))
    files = {"front": ("f.jpg", buf.tobytes()), "back": ("b.jpg", buf.tobytes())}
    settings = Settings(camera_enabled=False, visual_ocr=False, _env_file=None)
    with TestClient(create_app(settings)) as c:
        res = c.post("/mrz/leer", data={"documentType": "STUDENT_CARD"}, files=files)
    body = res.json()
    assert res.status_code == 200
    assert body["documentType"] == "STUDENT_CARD"  # lo que valida la regla del ticket 'STUDENT'
    assert body["suggestedCategory"] is None  # 'universitario' no esta en el enum del backend
    assert body["suggestedTicketTypeId"] is None and body["violations"] == []
    assert body["requiresManualEntry"] is False and body["status"] == "REVIEW"
    assert body["diagnostics"]["reasons"] == ["CARNET_UNIVERSITARIO_DETECTADO", "DATOS_NO_LEIDOS"]
    assert all(body["fields"][k] == "" for k in body["fields"])
