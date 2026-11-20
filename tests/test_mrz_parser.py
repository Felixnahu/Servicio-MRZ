"""Pruebas del parser MRZ con los especimenes oficiales de ICAO 9303 (datos ficticios)."""

from datetime import date

import pytest

from servicio_mrz.mrz import parse_mrz
from servicio_mrz.mrz.checkdigit import compute

TODAY = date(2026, 11, 20)

TD3_LINES = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C<3UTO6908061F9406236ZE184226B<<<<<14",
]
TD1_LINES = [
    "I<UTOD231458907<<<<<<<<<<<<<<<",
    "7408122F1204159UTO<<<<<<<<<<<6",
    "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
]
TD2_LINES = [
    "I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<",
    "D231458907UTO7408122F1204159<<<<<<<6",
]


def test_check_digit_known_values():
    assert compute("520727") == 3  # ejemplo de la norma ICAO 9303
    assert compute("L898902C<") == 3


def test_td3_passport_all_checks_pass():
    r = parse_mrz(TD3_LINES, today=TODAY)
    assert r is not None
    assert r.format == "TD3"
    assert r.checks == {
        "document_number": True,
        "birth_date": True,
        "expiry_date": True,
        "composite": True,
        "personal_number": True,
    }
    assert (r.document_code, r.issuing_state, r.nationality) == ("P", "UTO", "UTO")
    assert r.document_number == "L898902C"
    assert (r.surname, r.given_names) == ("ERIKSSON", "ANNA MARIA")
    assert r.birth_date == date(1969, 8, 6)
    assert r.sex == "F"


def test_td1_id_card_all_checks_pass():
    r = parse_mrz(TD1_LINES, today=TODAY)
    assert r is not None
    assert r.format == "TD1"
    assert all(r.checks.values()), r.checks
    assert r.document_number == "D23145890"
    assert r.birth_date == date(1974, 8, 12)
    assert (r.surname, r.given_names) == ("ERIKSSON", "ANNA MARIA")


def test_td2_all_checks_pass():
    r = parse_mrz(TD2_LINES, today=TODAY)
    assert r is not None
    assert r.format == "TD2"
    assert all(r.checks.values()), r.checks
    assert r.document_number == "D23145890"


def test_ocr_digit_letter_confusion_is_repaired_in_digit_fields():
    bad = TD3_LINES[:]
    bad[1] = "L898902C<3UTO69O8O61F94O6236ZE184226B<<<<<14"  # O en lugar de 0 en las fechas
    r = parse_mrz(bad, today=TODAY)
    assert r is not None and r.core_checks_ok
    assert r.birth_date == date(1969, 8, 6)


def test_document_number_letters_are_not_rewritten():
    r = parse_mrz(TD1_LINES, today=TODAY)
    assert r is not None and r.document_number.startswith("D")


def test_wrong_check_digit_is_reported_not_hidden():
    bad = TD3_LINES[:]
    bad[1] = "L898902C<3UTO6908062F9406236ZE184226B<<<<<14"  # fecha alterada, control igual
    r = parse_mrz(bad, today=TODAY)
    assert r is not None
    assert r.checks["birth_date"] is False
    assert not r.core_checks_ok


def test_short_lines_lost_trailing_fillers_still_parse():
    short = [TD3_LINES[0].rstrip("<"), TD3_LINES[1]]
    r = parse_mrz(short, today=TODAY)
    assert r is not None and r.surname == "ERIKSSON"


def test_birth_century_never_in_the_future():
    # yy=20 con hoy=2026: 2020 es pasado (nacido en 2020); yy=40 -> 1940, no 2040.
    from servicio_mrz.mrz.parser import _parse_date

    assert _parse_date("200101", birth=True, today=TODAY) == date(2020, 1, 1)
    assert _parse_date("400101", birth=True, today=TODAY) == date(1940, 1, 1)
    assert _parse_date("301231", birth=False, today=TODAY) == date(2030, 12, 31)


def test_invalid_date_is_none():
    from servicio_mrz.mrz.parser import _parse_date

    assert _parse_date("991345", birth=True, today=TODAY) is None


@pytest.mark.parametrize("lines", [[], ["ABC"], ["A" * 30, "B" * 30], ["X" * 10] * 3])
def test_garbage_returns_none(lines):
    assert parse_mrz(lines, today=TODAY) is None


def _td3_with(nationality: str, sex: str = "F") -> list[str]:
    """TD3 con TODOS los digitos de control correctos y los campos a medida."""
    from servicio_mrz.mrz import checkdigit

    def c(data: str) -> str:
        return str(checkdigit.compute(data))

    number, dob, exp, personal = "L898902C<", "690806", "940623", "ZE184226B<<<<<"
    line2 = (
        number
        + c(number)
        + nationality
        + dob
        + c(dob)
        + sex
        + exp
        + c(exp)
        + personal
        + c(personal)
    )
    line2 += c(number + c(number) + dob + c(dob) + exp + c(exp) + personal + c(personal))
    return ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", line2]


def test_garbage_with_valid_check_digits_is_rejected_when_nationality_does_not_exist():
    """Regresion: basura con 3 controles validos (nacionalidad EPA) se entregaba como lectura."""
    good = parse_mrz(_td3_with("UTO"))
    assert good and good.core_checks_ok
    bad = parse_mrz(_td3_with("EPA"))
    assert bad and all(bad.checks[k] for k in ("document_number", "birth_date", "expiry_date"))
    assert not bad.plausible and not bad.core_checks_ok


def test_implausible_sex_is_rejected():
    bad = parse_mrz(_td3_with("UTO", sex="X"))
    assert bad and not bad.core_checks_ok
