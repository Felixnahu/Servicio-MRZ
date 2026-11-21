"""Lugar de nacimiento -> isSamaipateno. Cajas sinteticas: sin datos de personas reales."""

from servicio_mrz import birthplace, contract
from servicio_mrz.visual import Box


def box(y: float, text: str, x: float = 100.0) -> Box:
    return Box(x, y, x + 300, y + 20, text)


def test_decide_is_asymmetric():
    assert birthplace.decide("SAMAIPATA") is True
    assert birthplace.decide("Samaipata - Florida") is True
    assert birthplace.decide("SAMAIPAIA") is True  # el OCR confunde letras
    assert birthplace.decide("SANTA CRUZ") is False
    assert birthplace.decide("") is None  # no se leyo nada: no se afirma "no"
    assert birthplace.decide("AB") is None


def test_new_card_value_below_label():
    boxes = [
        box(30, "LUGARDENACIMIENTO", 200),
        box(52, "SAMAIPATA", 200),
        box(110, "FECHA DE EMISION", 200),
    ]
    assert birthplace.from_new_boxes(boxes, 1000) is True


def test_new_card_other_city_and_missing_label():
    boxes = [box(30, "LUGAR DE NACIMIENTO", 200), box(52, "COCHABAMBA", 200)]
    assert birthplace.from_new_boxes(boxes, 1000) is False
    assert birthplace.from_new_boxes([box(52, "SAMAIPATA")], 1000) is None


def test_new_card_ignores_samaipata_elsewhere():
    """Una 'Samaipata' lejos de la etiqueta (p. ej. el lugar de emision) no cuenta."""
    boxes = [
        box(30, "LUGAR DE NACIMIENTO", 200),
        box(52, "LA PAZ", 200),
        box(300, "SAMAIPATA", 200),
    ]
    assert birthplace.from_new_boxes(boxes, 1000) is False


def test_old_card_place_on_line_below_date():
    date_line = box(100, "Nacido el 01 de Marzo de 2003 en", 40)
    assert birthplace.from_old_boxes([date_line, box(124, "SAMAIPATA - FLORIDA", 160)], date_line)
    other = [date_line, box(124, "SUCRE - CHUQUISACA", 160)]
    assert birthplace.from_old_boxes(other, date_line) is False
    assert birthplace.from_old_boxes([date_line], date_line) is None  # no se leyo el lugar


def test_suggested_category_samaipateno():
    assert contract.suggested_category("BOL", 30, True) == "samaipateno"
    assert contract.suggested_category("BOL", 30, False) == "nacional_adulto"
    assert contract.suggested_category("BOL", 30, None) == "nacional_adulto"


def test_target_place_can_be_changed_for_testing_and_is_restored():
    try:
        birthplace.set_target("Tupiza")
        assert birthplace.decide("TUPIZA - POTOSI") is True
        assert birthplace.decide("SAMAIPATA") is False  # ya no es el lugar buscado
        birthplace.set_target("  ")  # vacio: no cambia
        assert birthplace.TARGET == "TUPIZA"
    finally:
        birthplace.set_target("SAMAIPATA")
    assert birthplace.decide("SAMAIPATA") is True
