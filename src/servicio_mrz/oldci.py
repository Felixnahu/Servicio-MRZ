"""Lectura por PLANTILLA de la cedula boliviana antigua (la de 1975 y la de 2020), sin MRZ.

Las dos versiones comparten formato; la unica diferencia es que la de 1975 trae un codigo de
barras y la de 2020 un QR en el anverso. Los datos estan repartidos entre las dos caras:

- REVERSO: "A: <nombre completo>" (una sola linea) y "Nacido el 15 de Marzo de 1985".
- ANVERSO: "No. 1234567" (numero de carnet, en rojo claro, abajo a la izquierda).

Estrategia (medida en 12 fotos de las dos versiones, ver tests/measure_old_ci.py):

1. El reverso se lee entero: recortar una banda con el nombre y la fecha daba peores resultados
   que la tarjeta completa (la fecha fallaba en 2 de 5). Se parsea por patron de fecha, no por
   etiqueta: el OCR deforma "Nacido" y "CERTIFICA" a menudo.
2. Si se obtiene la fecha y queda tiempo, se recorta la zona del numero del anverso. El numero es
   opcional y no debe consumir el plazo antes de leer el reverso.

Igual que el OCR de campos impresos, NO hay digitos de control: el resultado va marcado `REVIEW` y
solo se devuelve si se leyo una fecha de nacimiento real. El numero se lee en ~60 % de los
anversos (el rojo claro sobre beige a veces ni se detecta): si falta, se avisa y se completa a mano.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import cv2
import numpy as np

from servicio_mrz import birthplace, orientation, student, visual
from servicio_mrz.visual import Box

# Zona del anverso con "No. 1234567": ((y0, y1), (x0, x1)) como fraccion de la tarjeta enderezada.
FRONT_REGION = ((0.55, 0.80), (0.0, 0.6))
_NOT_NAME = ("PERTENECE", "IMPRESION", "CERTIFICA", "FIRMA", "SERVICIO", "IDENTIFICACION")
_MIN_YEAR = 1900


@dataclass
class OldCiRead:
    born: date | None = None
    full_name: str = ""
    number: str = ""
    samaipateno: bool | None = None
    student: bool = False  # la tarjeta es un carnet universitario (no una cedula antigua)

    def merge(self, other: OldCiRead) -> None:
        self.born = self.born or other.born
        if self.samaipateno is None:
            self.samaipateno = other.samaipateno
        self.full_name = self.full_name or other.full_name
        self.number = self.number or other.number


def _strip_label(text: str) -> str:
    """Quita un 'A:' inicial si el OCR lo pego al nombre ('A: PAULA ...')."""
    words = text.split()
    if words and re.fullmatch(r"[AÁ][:.;]?", words[0].upper()):
        words = words[1:]
    return " ".join(words)


def parse_back(boxes: list[Box], width: float, today: date | None = None) -> OldCiRead | None:
    """Nombre y fecha de nacimiento del reverso a partir de las cajas de la tarjeta."""
    # La lectura puede probarse tambien en una pagina de pasaporte sin MRZ detectada. Exigir una
    # senal propia del reverso antiguo evita tomar su fecha de nacimiento como carnet antiguo.
    keys = [visual._key(b.text) for b in boxes]
    if not any(
        "CERTIF" in key or "NACID" in key or "NACLD" in key
        for key in keys
    ):
        return None
    dated = []
    for b in boxes:
        # En el mes escrito con letras, RapidOCR puede leer la O final como cero ("Marz0").
        # Se corrige solo ese caso; los digitos del dia y del ano permanecen intactos.
        date_text = re.sub(r"(?<=[A-Za-z])0\b", "O", b.text)
        born = visual.parse_date(date_text, today)
        if born is not None and born.year >= _MIN_YEAR:
            dated.append((b, born))
    if not dated:
        return None
    line, born = min(dated, key=lambda item: item[0].cy)  # la fecha de nacimiento es la 1.a
    candidates = []
    for b in boxes:
        if b.cy >= line.cy - 0.3 * line.h or b.x0 < 0.12 * width:
            continue  # debajo de la fecha, o la etiqueta 'A:' del margen
        text = _strip_label(visual._name(b.text))
        key = visual._key(b.text)
        if any(w in key for w in _NOT_NAME) or any(c.isdigit() for c in b.text):
            continue
        if len(text.split()) >= 2 and len(text) >= 6:
            candidates.append((b, text))
    name = max(candidates, key=lambda item: item[0].cy)[1] if candidates else ""
    return OldCiRead(
        born=born, full_name=name, samaipateno=birthplace.from_old_boxes(boxes, line)
    )


def parse_front_number(boxes: list[Box]) -> str:
    """Numero de carnet: la primera cifra de 5 a 9 digitos de la zona 'No.'."""
    for b in sorted(boxes, key=lambda b: (b.y0, b.x0)):
        fixed = b.text.upper().translate(visual._DIGIT_FIX)
        for m in re.finditer(r"\d[\d ]{3,10}\d", fixed):
            digits = re.sub(r"\D", "", m[0])
            if 5 <= len(digits) <= 9:
                return digits
    return ""


def _front_number(engine: Any, card: np.ndarray) -> str:
    """Lee solo la zona del numero. Prueba color si el canal verde pierde el rojo claro."""
    h, w = card.shape[:2]
    (y0, y1), (x0, x1) = FRONT_REGION
    region = card[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
    green = region[:, :, 1]
    green = cv2.resize(green, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    boxes, _, _ = visual.read_boxes(
        engine, cv2.cvtColor(green, cv2.COLOR_GRAY2BGR), use_cls=False
    )
    number = parse_front_number(boxes)
    if number:
        return number
    # El reverso no tiene "No.". Evitar una segunda inferencia en cada reverso ahorra tiempo
    # dentro del plazo de la app, especialmente en el totem.
    if not any(b.key in {"NO", "N0"} for b in boxes):
        return ""
    color = cv2.resize(region, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    boxes, _, _ = visual.read_boxes(engine, color, use_cls=False)
    return parse_front_number(boxes)


def _back(engine: Any, card: np.ndarray) -> OldCiRead | None:
    boxes, width, _ = visual.read_boxes(engine, card, use_cls=False)
    read = parse_back(boxes, width)
    if read is None and student.detect(boxes):
        # Mismo OCR, sin coste extra: un carnet universitario no tiene fecha ni formato antiguo.
        return OldCiRead(student=True)
    return read


def _upright_card(engine: Any, img: np.ndarray) -> np.ndarray:
    """Tarjeta enderezada con el texto del derecho.

    Si el clasificador de giro dice que esta al reves (0.1 s), se gira la FOTO y se vuelve a
    localizar: el localizador recorta mejor una tarjeta del derecho (con la tarjeta enderezada y
    girada despues la fecha del reverso se perdia en 2 de 5 carnets). Evita gastar una pasada
    completa de OCR (4-9 s en el totem) leyendo primero la tarjeta girada.
    """
    card = visual._prepare(img)
    upside = orientation.upside_down(engine, card)
    if upside is None and orientation.sideways(engine, card):
        # tarjeta puesta de lado: casi todo el texto es vertical. Se gira 90 grados y se decide
        # entre las dos direcciones con el mismo clasificador de giro.
        turned = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
        if orientation.upside_down(engine, turned):
            return cv2.rotate(turned, cv2.ROTATE_180)
        return turned
    if not upside:
        return card
    flipped = cv2.rotate(img, cv2.ROTATE_180)
    again = visual._prepare(flipped)
    if orientation.upside_down(engine, again) is False:
        return again
    return cv2.rotate(card, cv2.ROTATE_180)


def read_old_ci(
    engine: Any,
    images: list[np.ndarray],
    deadline: float | None = None,
    detect_student: bool = True,
) -> dict[str, Any] | None:
    """Datos de una cedula antigua a partir de sus fotos (frente y/o reverso, en cualquier orden).

    Si la tarjeta resulta ser un carnet universitario (`detect_student`), devuelve solo la
    categoria sugerida (ver `student.py`) con los datos vacios para completarlos a mano.

    `deadline` (time.monotonic) corta el trabajo entre pasos: en un equipo lento o saturado esta
    lectura puede pasar de 40 s y la app abandona a los 45. None si no se logra la fecha de
    nacimiento (o se agota el tiempo): el llamador pasa al respaldo.
    """

    def late() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    cards = []
    for img in images:
        if late():
            break
        cards.append(_upright_card(engine, img))
    total = OldCiRead()
    # La fecha del reverso es obligatoria; buscar antes el numero opcional agotaba el plazo en el
    # totem. Se prueba primero la ultima foto (la app pide el reverso en 2.o lugar).
    # Orden de las pasadas de OCR: las DOS fotos como estan y solo despues las giradas 180 grados.
    # Cada pasada tarda ~7 s en el totem y el plazo minimo es corto: girar primero la 1.a foto
    # gastaba el plazo antes de mirar la otra cara (donde estaba, p. ej., "UNIVERSITARIO").
    back_idx = None
    for rotated in (False, True):
        for i in reversed(range(len(cards))):
            if total.born is not None:
                break
            if late():
                break
            card = cv2.rotate(cards[i], cv2.ROTATE_180) if rotated else cards[i]
            part = _back(engine, card)
            if part is not None and part.student:
                if detect_student:
                    return student.result()
                continue
            if part is not None:
                total.merge(part)
                back_idx = i
        if total.born is not None or late():
            break
    if total.born is None:
        return None
    # El numero se lee solo si queda margen y nunca de la cara ya reconocida como reverso.
    for i, card in enumerate(cards):
        if i == back_idx or late():
            continue
        total.number = _front_number(engine, card)
        if total.number:
            break
    words = total.full_name.split()
    cut = max(len(words) - 2, 1) if len(words) > 1 else len(words)  # 2 ultimos = apellidos
    return visual._result(
        "CI_BO",
        " ".join(words[:cut]),
        " ".join(words[cut:]),
        total.number,
        "BOL",
        total.born,
        "ci-old",
        extra_reasons=("DIVISION_DE_NOMBRES_ESTIMADA",) if len(words) > 1 else (),
        samaipateno=total.samaipateno,
    )
