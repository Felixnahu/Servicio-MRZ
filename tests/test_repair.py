"""Correccion de UNA cifra dudosa con los digitos de control. Especimenes ICAO (datos ficticios)."""

import numpy as np

from servicio_mrz import celldecode, mrzread
from servicio_mrz.mrz import parse_mrz

TD1 = [
    "I<UTOD231458907<<<<<<<<<<<<<<<",
    "7408122F1204159UTO<<<<<<<<<<<6",
    "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
]
TD3 = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C<3UTO6908061F9406236ZE184226B<<<<<14",
]
A = celldecode.ALPHABET


def _scores(lines, doubt=None):
    """0.9 en el caracter leido; `doubt` = {(linea, celda): (alt, p_alt, p_leido)}."""
    out = []
    for text in lines:
        probs = np.full((len(text), len(A)), 0.001)
        for ci, ch in enumerate(text):
            probs[ci, A.index(ch)] = 0.9
        out.append(probs)
    for (li, ci), (alt, p_alt, p_read) in (doubt or {}).items():
        read = lines[li][ci]
        out[li][ci, A.index(read)] = p_read
        out[li][ci, A.index(alt)] = p_alt
    return out


def _corrupt(lines, li, ci, wrong):
    fixed = list(lines)
    fixed[li] = fixed[li][:ci] + wrong + fixed[li][ci + 1 :]
    return fixed


def test_a_doubtful_digit_is_fixed_when_every_check_then_passes():
    wrong = _corrupt(TD3, 1, 3, "9")  # el numero de documento: 8 leido como 9
    assert parse_mrz(wrong).core_checks_ok is False
    scores = _scores(wrong, {(1, 3): ("8", 0.35, 0.40)})  # la correcta era la 2.a opcion
    fixed = mrzread.repair_checks(wrong, scores)
    assert fixed is not None and fixed.repaired == 1 and fixed.core_checks_ok
    assert fixed.document_number == "L898902C"
    assert fixed.checks["composite"] is True


def test_no_repair_when_the_right_digit_was_not_a_serious_alternative():
    wrong = _corrupt(TD3, 1, 3, "9")
    scores = _scores(wrong, {(1, 3): ("8", 0.05, 0.85)})  # el OCR estaba seguro: no se toca
    assert mrzread.repair_checks(wrong, scores) is None


def test_no_repair_when_the_alternative_does_not_validate():
    wrong = _corrupt(TD3, 1, 3, "9")
    scores = _scores(wrong, {(1, 3): ("7", 0.40, 0.45)})  # la 2.a opcion tampoco cuadra
    assert mrzread.repair_checks(wrong, scores) is None


def test_a_valid_reading_is_never_touched():
    assert mrzread.repair_checks(TD3, _scores(TD3)) is None
    assert mrzread.repair_checks(TD1, _scores(TD1)) is None


def test_td1_document_number_is_repaired_too():
    wrong = _corrupt(TD1, 0, 6, "0")  # D231458907: el 2 del numero leido como 0
    scores = _scores(wrong, {(0, 6): ("2", 0.33, 0.38)})
    assert parse_mrz(wrong).core_checks_ok is False
    fixed = mrzread.repair_checks(wrong, scores)
    assert fixed is not None and fixed.repaired == 1 and fixed.document_number == "D23145890"


def test_parse_helper_returns_the_plain_result_when_nothing_can_be_fixed():
    wrong = _corrupt(TD3, 1, 3, "9")
    result = mrzread._parse(wrong, _scores(wrong))  # sin celdas dudosas
    assert result is not None and result.core_checks_ok is False and result.repaired == 0
