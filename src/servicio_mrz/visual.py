"""Lectura de los campos IMPRESOS del documento (OCR por etiqueta), para cuando no hay MRZ legible.

Sirve para el anverso de la cedula 2023, la pagina de datos de un pasaporte y la cedula antigua
(sin MRZ). Se hace OCR general sobre la tarjeta enderezada y, para cada campo, se localiza su
etiqueta (NOMBRES, APELLIDOS, FECHA DE NACIMIENTO...) y se toma el texto que esta debajo.

Importante: a diferencia de la MRZ NO hay digitos de control. Un digito mal leido da un dato
equivocado sin que nada lo detecte; por eso el resultado va marcado `source: "ocr"` y
`status: "REVIEW"`, y solo se devuelve si la fecha de nacimiento es una fecha real y plausible.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any

import cv2
import numpy as np

from servicio_mrz.contract import ADULT_AGE, age_at, today_in_bolivia
from servicio_mrz.mrz.countries import VALID_CODES

_MONTHS = {
    "ENE": 1, "ENERO": 1, "JAN": 1, "FEB": 2, "FEBRERO": 2, "MAR": 3, "MARZO": 3, "ABR": 4,
    "ABRIL": 4, "APR": 4, "MAY": 5, "MAYO": 5, "JUN": 6, "JUNIO": 6, "JUL": 7, "JULIO": 7,
    "AGO": 8, "AGOSTO": 8, "AUG": 8, "SEP": 9, "SET": 9, "SEPTIEMBRE": 9, "SETIEMBRE": 9,
    "OCT": 10, "OCTUBRE": 10, "NOV": 11, "NOVIEMBRE": 11, "DIC": 12, "DICIEMBRE": 12, "DEC": 12,
}  # fmt: skip
_DIGIT_FIX = str.maketrans("OoIlLSBZ", "00111582")


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return max(self.y1 - self.y0, 1.0)

    @property
    def key(self) -> str:
        return _key(self.text)


def _key(text: str) -> str:
    """Mayusculas, sin tildes ni signos: para reconocer etiquetas aunque el OCR las deforme."""
    text = unicodedata.normalize("NFKD", text.upper())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9]", "", text)


def _name(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.upper())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^A-Z ]", " ", text).split())


def parse_date(text: str, today: date | None = None) -> date | None:
    """dd/mm/aaaa, dd/MES/MON/aaaa (pasaporte) o 'd de Enero de aaaa' (cedula antigua)."""
    today = today or today_in_bolivia()
    t = unicodedata.normalize("NFKD", text.upper())
    t = "".join(c for c in t if not unicodedata.combining(c))
    day = month = year = None
    m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})", t.translate(_DIGIT_FIX))
    if m:
        day, month, year = int(m[1]), int(m[2]), int(m[3])
    else:
        m = re.search(r"([0-9OIlS]{1,2})\s*(?:DE\s*)?[/\-. ]*\s*([A-Z]{3,10})\s*(?:DE|[/\-. ])*\s*"
                      r"(?:[A-Z]{3}\s*[/\-. ]*\s*)?([0-9OIlSBZ]{4})", t)  # fmt: skip
        if m and m[2] in _MONTHS:
            day, month, year = (
                int(m[1].translate(_DIGIT_FIX)),
                _MONTHS[m[2]],
                int(m[3].translate(_DIGIT_FIX)),
            )
    if day is None:
        return None
    try:
        found = date(year, month, day)
    except ValueError:
        return None
    return found if 1900 <= found.year <= today.year and found <= today else None


def _below(label: Box, boxes: list[Box], width: float, height: float, skip: set[str]) -> Box | None:
    """Caja justo debajo de la etiqueta (alineada a su izquierda), que no sea otra etiqueta."""
    best, best_dy = None, 0.0
    for b in boxes:
        if b is label or b.key in skip or any(k in b.key for k in skip if len(k) > 5):
            continue
        dy = b.y0 - label.y1
        if dy < -0.5 * label.h or dy > 0.12 * height or abs(b.x0 - label.x0) > 0.07 * width:
            continue
        if best is None or dy < best_dy:
            best, best_dy = b, dy
    return best


def _below_lines(
    label: Box, boxes: list[Box], width: float, height: float, skip: set[str], lines: int = 1
) -> str:
    """Texto debajo de la etiqueta; un nombre largo puede ocupar una segunda linea."""
    parts: list[str] = []
    anchor = label
    for _ in range(lines):
        nxt = _below(anchor, boxes, width, height, skip)
        # la 2a linea debe quedar pegada a la 1a (hueco < 1 alto de linea), no ser otro campo
        if nxt is None or (parts and nxt.y0 - anchor.y1 > 0.9 * anchor.h):
            break
        parts.append(nxt.text.strip())
        anchor = nxt
    return " ".join(parts)


def _right_of(label: Box, boxes: list[Box], width: float) -> Box | None:
    cands = [
        b
        for b in boxes
        if b is not label
        and b.x0 >= label.x1 - 0.01 * width
        and abs(b.cy - label.cy) < 0.7 * label.h
    ]
    return min(cands, key=lambda b: b.x0) if cands else None


def _find(boxes: list[Box], *needles: str) -> Box | None:
    for b in boxes:
        if any(n in b.key for n in needles):
            return b
    return None


_LABELS = {
    "NOMBRES", "APELLIDOS", "SERIE", "SECCION", "FECHADENACIMIENTO", "FECHADEEMISION",
    "FECHADEEXPIRACION", "SURNAMES", "GIVENNAMES", "NACIONALIDAD", "NATIONALITY", "SEXO",
    "LUGARDENACIMIENTO", "DATEOFBIRTH", "PLACEOFBIRTH", "DATEOFISSUE", "DATEOFEXPIRY",
    "PASAPORTE", "PASSPORT", "FIRMA", "SIGNATURE", "TIPO", "CODIGODELPAIS",
}  # fmt: skip


def _orient(boxes: list[Box], width: float, height: float) -> list[Box]:
    """Si la tarjeta esta boca abajo (encabezado abajo), gira las coordenadas 180 grados."""
    head = _find(
        boxes, "ESTADOPLURINACIONAL", "SERVICIOGENERAL", "PASAPORTE", "PASSPORT", "CEDULADE"
    )
    if head is not None and head.cy > 0.6 * height:
        return [
            Box(width - b.x1, height - b.y1, width - b.x0, height - b.y0, b.text) for b in boxes
        ]
    return boxes


def _result(
    doc_type: str,
    first: str,
    last: str,
    number: str,
    nationality: str,
    born: date,
    layout: str,
    extra_reasons: tuple[str, ...] = (),
) -> dict[str, Any]:
    today = today_in_bolivia()
    age = age_at(born, today)
    return {
        "documentType": doc_type,
        "documentModel": layout,
        "fields": {
            "firstNames": first,
            "lastNames": last,
            "documentNumber": number,
            "nationality": nationality,
            "birthDate": born.isoformat(),
        },
        "age": age,
        "isAdult": age >= ADULT_AGE,
        "isSamaipateno": None,
        "suggestedCategory": None,
        "categoryMatches": None,
        "source": "ocr",
        "requiresManualEntry": False,
        "status": "REVIEW",
        "reasons": [
            "SIN_DIGITOS_DE_CONTROL",
            *(["NOMBRES_NO_LEIDOS"] if not (first or last) else []),
            *(["NUMERO_NO_LEIDO"] if not number else []),
            *extra_reasons,
        ],
    }


def _value(box: Box | None) -> str:
    return box.text.strip() if box else ""


def parse_fields(boxes: list[Box], width: float, height: float) -> dict[str, Any] | None:
    """Campos del documento a partir de las cajas de texto del OCR. None si no hay fecha valida."""
    boxes = _orient(boxes, width, height)
    keys = " ".join(b.key for b in boxes)
    is_passport = "SURNAMES" in keys or "GIVENNAMES" in keys or "PASSPORT" in keys
    if "NACIDOEL" in keys and "CERTIFICA" in keys:
        return _old_ci(boxes, width, height)
    skip = _LABELS
    dob_label = _find(boxes, "FECHADENACIMIENTO", "DATEOFBIRTH")
    born = parse_date(_value(_below(dob_label, boxes, width, height, skip))) if dob_label else None
    if born is None:
        return None
    surname_label = _find(boxes, "APELLIDOS", "SURNAMES")
    given_label = _find(boxes, "NOMBRES", "GIVENNAMES")
    last = _name(_below_lines(surname_label, boxes, width, height, skip)) if surname_label else ""
    first = _name(_below_lines(given_label, boxes, width, height, skip)) if given_label else ""
    if is_passport:
        serial = _find(boxes, "SERIE", "SERIAL")
        number = (
            re.sub(r"[^A-Z0-9]", "", _key(_value(_below(serial, boxes, width, height, skip))))
            if serial
            else ""
        )
        code = _find(boxes, "CODIGODELPAIS", "COUNTRYCODE")
        nat = _key(_value(_below(code, boxes, width, height, skip))) if code else ""
        nat = nat if nat in VALID_CODES else ("BOL" if "BOLIVIANA" in keys else "")
        return _result("PASSPORT", first, last, number, nat, born, "passport-viz")
    number = ""
    for b in boxes:  # "N° 1234567" (arriba a la derecha)
        m = re.search(
            r"N\s*[°º0oO]?\s*[:.]?\s*(\d[\d .]{4,11})", b.text.upper().translate(_DIGIT_FIX)
        )
        if m and b.x0 > 0.55 * width and b.cy < 0.35 * height:
            number = re.sub(r"\D", "", m[1])
            break
    if not (first or last):
        return None
    return _result("CI_BO", first, last, number, "BOL", born, "ci-2023-front")


def _old_ci(boxes: list[Box], width: float, height: float) -> dict[str, Any] | None:
    """Reverso de la cedula antigua: 'A: nombre completo' y 'Nacido el 5 de Marzo de 1985'."""
    nacido = _find(boxes, "NACIDOEL")
    born = parse_date(nacido.text) if nacido else None
    if born is None:
        return None
    a = next(
        (b for b in boxes if b.key in ("A", "A1") or b.key.startswith("A") and len(b.key) <= 2),
        None,
    )
    full = _name(_value(_right_of(a, boxes, width))) if a else ""
    words = full.split()
    cut = max(len(words) - 2, 1) if len(words) > 1 else len(words)  # dos apellidos al final
    number = ""
    for b in boxes:
        m = re.search(
            r"(\d[\d ]{5,10}\d)\s*[-\s]?\s*\d{0,2}\s*-?\s*[A-Z]?\d?", b.text.translate(_DIGIT_FIX)
        )
        if m and b.cy > 0.75 * height and b.x0 < 0.4 * width:
            number = re.sub(r"\D", "", m[1])[:8]
            break
    # el reverso trae el nombre completo en una sola linea: la division nombres/apellidos es una
    # estimacion (convencion: los dos ultimos terminos son los apellidos) y se avisa
    return _result(
        "CI_BO",
        " ".join(words[:cut]),
        " ".join(words[cut:]),
        number,
        "BOL",
        born,
        "ci-old-back",
        extra_reasons=("DIVISION_DE_NOMBRES_ESTIMADA",) if len(words) > 1 else (),
    )


WORD_GAP = (
    0.6  # hueco entre caracteres, en anchos medios de caracter, a partir del cual hay un espacio
)


def _with_spaces(text: str, char_boxes: list, chars: list) -> str:
    """Reconstruye los espacios que el reconocedor omite, mirando el hueco entre caracteres.

    El OCR a veces une palabras en negrita ('MARIALUISA'): su texto no trae el espacio, pero las
    cajas por caracter si muestran el hueco.
    """
    if not chars or len(chars) != len(char_boxes):
        return text
    spans = [(min(p[0] for p in b), max(p[0] for p in b)) for b in char_boxes]
    widths = [x1 - x0 for (x0, x1), c in zip(spans, chars, strict=True) if c.strip()]
    if not widths:
        return text
    limit = WORD_GAP * float(np.median(widths))
    out = []
    for k, ch in enumerate(chars):
        if k and ch.strip() and chars[k - 1].strip() and spans[k][0] - spans[k - 1][1] > limit:
            out.append(" ")
        out.append(ch)
    return "".join(out)


def read_boxes(engine: Any, img: np.ndarray) -> tuple[list[Box], float, float]:
    """OCR general (deteccion + reconocimiento) y cajas de texto en coordenadas de `img`."""
    result, _ = engine(img, return_word_box=True)
    boxes = []
    for item in result or []:
        points, text = item[0], item[1]
        if len(item) >= 5:
            text = _with_spaces(text, item[3], item[4])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        boxes.append(Box(min(xs), min(ys), max(xs), max(ys), text))
    return boxes, float(img.shape[1]), float(img.shape[0])


def _prepare(img: np.ndarray) -> np.ndarray:
    """Tarjeta enderezada si se localiza; si ya es un recorte, entero; si no, reducida."""
    from servicio_mrz import cardfinder

    try:
        located = cardfinder.locate_card(img)
    except Exception:
        located = None
    if located is not None:
        return cardfinder.warp_card(img, located).image
    whole = cardfinder.whole_image_as_card(img)
    if whole is not None:
        return whole.image
    scale = min(1.0, 1024 / max(img.shape[:2]))
    return img if scale == 1.0 else cv2.resize(img, None, fx=scale, fy=scale)


def read_image(engine: Any, img: np.ndarray) -> dict[str, Any] | None:
    """Campos impresos de una foto de documento, o None si no se obtiene una fecha valida."""
    boxes, width, height = read_boxes(engine, _prepare(img))
    return parse_fields(boxes, width, height)
