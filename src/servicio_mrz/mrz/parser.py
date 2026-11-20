"""Parser de MRZ ICAO 9303 (TD1 3x30, TD2 2x36, TD3 2x44) con verificacion de digitos de control.

Recibe lineas ya leidas por OCR (con posibles confusiones 0/O, 1/I, etc.), corrige por tipo de
campo (digitos vs letras) y verifica los digitos de control. Nunca inventa datos: si un digito de
control no cuadra, lo dice en `checks` y deja la decision a quien llama.

El numero de documento NO se corrige: puede llevar letras y digitos mezclados.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from servicio_mrz.mrz import checkdigit
from servicio_mrz.mrz.countries import VALID_CODES

TD1, TD2, TD3 = "TD1", "TD2", "TD3"
_SIZES = {3: (30,), 2: (36, 44)}  # lineas -> largos validos
_LEN_TOLERANCE = 3  # el OCR suele perder '<' del final

_TO_DIGIT = str.maketrans("OQDILZSBG", "000112586")
_TO_ALPHA = str.maketrans("01258", "OIZSB")


@dataclass(frozen=True)
class MrzResult:
    format: str
    raw_lines: list[str]
    document_code: str
    issuing_state: str
    document_number: str
    nationality: str
    birth_date: date | None
    expiry_date: date | None
    sex: str
    surname: str
    given_names: str
    optional_data: str
    checks: dict[str, bool] = field(default_factory=dict)
    fused_frames: int = 0  # >0: lectura obtenida fusionando las probabilidades de N cuadros

    @property
    def plausible(self) -> bool:
        """Forma coherente, no solo digitos de control que cuadran por azar.

        Una lectura basura valida 3 controles con probabilidad ~1/1000 por intento; con muchas
        hipotesis/variantes/cuadros eso ocurre (medido: un pasaporte 'leido' como cedula con
        nacionalidad inexistente). Pais emisor y nacionalidad deben ser codigos reales, etc.
        """
        return bool(
            self.issuing_state in VALID_CODES
            and self.nationality in VALID_CODES
            and self.sex in ("M", "F", "<")
            and self.document_code[:1] in ("P", "I", "A", "C", "V")
            and self.birth_date is not None
            and self.expiry_date is not None
            and self.birth_date < self.expiry_date
            and self.document_number
            and self.surname.replace(" ", "").isalpha()
            and (not self.given_names or self.given_names.replace(" ", "").isalpha())
        )

    @property
    def core_checks_ok(self) -> bool:
        """Numero, nacimiento y vencimiento (los datos que la app usa) y forma plausible."""
        return self.plausible and all(
            self.checks.get(k, False) for k in ("document_number", "birth_date", "expiry_date")
        )


def normalize_line(line: str) -> str:
    line = re.sub(r"\s+", "", line.upper())
    return line.translate(str.maketrans("«‹({", "<<<<"))


def _fit(line: str, length: int) -> str:
    return line[:length].ljust(length, "<")


def _digits(s: str) -> str:
    return s.translate(_TO_DIGIT)


def _alpha(s: str) -> str:
    return s.translate(_TO_ALPHA)


def _parse_date(yymmdd: str, *, birth: bool, today: date) -> date | None:
    """Nacimiento: el siglo mas reciente que no quede en el futuro. Vencimiento: siempre 20yy."""
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:])
    try:
        if not birth:
            return date(2000 + yy, mm, dd)
        candidate = date(2000 + yy, mm, dd)
        return candidate if candidate <= today else date(1900 + yy, mm, dd)
    except ValueError:
        return None


def _names(zone: str) -> tuple[str, str]:
    surname, _, given = zone.partition("<<")

    def clean(s: str) -> str:
        return " ".join(_alpha(s).replace("<", " ").split())

    return clean(surname), clean(given)


def _document_number(number: str, check: str, extension: str) -> tuple[str, bool]:
    """Numero de documento y si su digito de control cuadra.

    Si el numero mide mas de 9 caracteres, el control es '<' y el resto viaja al inicio del campo
    opcional, seguido de su propio digito de control (ICAO 9303).
    """
    if check != "<":
        return number.rstrip("<"), checkdigit.verify(number, check)
    tail = extension.split("<")[0]
    if len(tail) < 2:
        return number.rstrip("<"), False
    full = number + tail[:-1]
    return full, checkdigit.verify(full, tail[-1])


def _td1(lines: list[str], today: date) -> MrzResult:
    l1, l2, l3 = (_fit(x, 30) for x in lines)
    l1 = _alpha(l1[:2]) + _alpha(l1[2:5]) + l1[5:14] + _digits(l1[14]) + l1[15:]
    l2 = (
        _digits(l2[:7])
        + l2[7]
        + _digits(l2[8:15])
        + _alpha(l2[15:18])
        + l2[18:29]
        + _digits(l2[29])
    )
    number, number_ok = _document_number(l1[5:14], l1[14], l1[15:30])
    surname, given = _names(l3)
    composite = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
    return MrzResult(
        format=TD1,
        raw_lines=[l1, l2, l3],
        document_code=l1[0:2].rstrip("<"),
        issuing_state=l1[2:5],
        document_number=number,
        nationality=l2[15:18],
        birth_date=_parse_date(l2[0:6], birth=True, today=today),
        expiry_date=_parse_date(l2[8:14], birth=False, today=today),
        sex=l2[7],
        surname=surname,
        given_names=given,
        optional_data=(l1[15:30] + l2[18:29]).replace("<", " ").strip(),
        checks={
            "document_number": number_ok,
            "birth_date": checkdigit.verify(l2[0:6], l2[6]),
            "expiry_date": checkdigit.verify(l2[8:14], l2[14]),
            "composite": checkdigit.verify(composite, l2[29]),
        },
    )


def _td2_td3(lines: list[str], today: date, *, length: int) -> MrzResult:
    l1, l2 = (_fit(x, length) for x in lines)
    last = length - 1
    opt_end = 42 if length == 44 else last  # TD3: personal number 28..41 + control en 42
    l1 = _alpha(l1[:2]) + _alpha(l1[2:5]) + l1[5:]
    l2 = (
        l2[:9]
        + _digits(l2[9])
        + _alpha(l2[10:13])
        + _digits(l2[13:20])
        + l2[20]
        + _digits(l2[21:28])
        + l2[28:last]
        + _digits(l2[last])
    )
    number, number_ok = _document_number(l2[0:9], l2[9], l2[28:opt_end])
    surname, given = _names(l1[5:])
    checks = {
        "document_number": number_ok,
        "birth_date": checkdigit.verify(l2[13:19], l2[19]),
        "expiry_date": checkdigit.verify(l2[21:27], l2[27]),
        "composite": checkdigit.verify(l2[0:10] + l2[13:20] + l2[21:last], l2[last]),
    }
    optional = l2[28:opt_end]
    if length == 44 and optional.strip("<") and l2[42] != "<":
        checks["personal_number"] = checkdigit.verify(optional, l2[42])
    return MrzResult(
        format=TD3 if length == 44 else TD2,
        raw_lines=[l1, l2],
        document_code=l1[0:2].rstrip("<"),
        issuing_state=l1[2:5],
        document_number=number,
        nationality=l2[10:13],
        birth_date=_parse_date(l2[13:19], birth=True, today=today),
        expiry_date=_parse_date(l2[21:27], birth=False, today=today),
        sex=l2[20],
        surname=surname,
        given_names=given,
        optional_data=optional.replace("<", " ").strip(),
        checks=checks,
    )


def parse_mrz(lines: list[str], *, today: date | None = None) -> MrzResult | None:
    """Parsea 2 o 3 lineas MRZ. None si la forma no corresponde a ningun formato ICAO."""
    today = today or date.today()
    cleaned = [normalize_line(x) for x in lines if x.strip()]
    sizes = _SIZES.get(len(cleaned))
    if not sizes:
        return None
    longest = max(len(x) for x in cleaned)
    size = min(sizes, key=lambda s: abs(s - longest))
    if abs(size - longest) > _LEN_TOLERANCE:
        return None
    return _td1(cleaned, today) if size == 30 else _td2_td3(cleaned, today, length=size)
