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
    assert not student.detect(boxes("CARRERA", "ESTUDIANTE"))  # apellido + ocupacion


def test_the_three_weak_words_together_do_count():
    assert student.detect(boxes("ESTUDIANTE:", "CARRERA:", "FACULTAD DE INGENIERIA"))


def test_result_has_the_contract_fields_and_asks_for_manual_entry():
    r = student.result()
    assert r["documentType"] == "STUDENT_CARD" and r["suggestedCategory"] == "universitario"
    assert r["requiresManualEntry"] is True and r["fields"]["birthDate"] == ""
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
    assert res is not None and res["suggestedCategory"] == "universitario"
    assert len(calls) == 1  # se detiene en la primera lectura
    assert oldci.read_old_ci(object(), [img, img], detect_student=False) is None
