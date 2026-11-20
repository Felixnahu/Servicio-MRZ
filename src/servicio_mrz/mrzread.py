"""Lee la MRZ de una tarjeta ya enderezada: segmenta lineas, formula hipotesis y decide con los
digitos de control.

Hipotesis (la segmentacion puede perder una linea o la tarjeta ser un pasaporte):
  - 3 lineas halladas        -> TD1 (cedula)
  - 2 lineas halladas        -> TD3 (pasaporte, 44 celdas) o TD1 con una linea perdida
                                 (la linea faltante se prueba arriba y abajo, por espaciado)
Cada hipotesis se decodifica celda por celda y se parsea; gana la que valida los controles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from servicio_mrz import celldecode, mrzlines, preprocess
from servicio_mrz.mrz import MrzResult, parse_mrz
from servicio_mrz.mrzlines import LineSegment

# Una MRZ legible da >= 0.84 con la 1a variante y el texto normal <= 0.47 (medido). Por debajo de
# este umbral no se prueban mas variantes: no hay MRZ legible y el OCR visual toma el relevo.
NO_MRZ_CONFIDENCE = 0.55  # medido: bajo esto la 1a variante lee basura y las demas tambien
PREFERRED_BONUS = 1.2  # a igual lectura, se prefiere el formato que sugiere la forma de la tarjeta
MIN_CONFIDENCE = 0.45  # por debajo, el reconocedor estaba "adivinando" (medido: ~0.6 vs ~0.2)


@dataclass(frozen=True)
class Observation:
    """Probabilidades por celda de UN cuadro, para fusionarlas con las de otros cuadros."""

    lines: int  # 3 (TD1) o 2 (TD3)
    scores: tuple[np.ndarray, ...]  # por linea: (celdas, len(ALPHABET))
    confidence: float


@dataclass(frozen=True)
class CardRead:
    result: MrzResult | None
    segments: tuple[LineSegment, ...]  # lineas reales halladas (para dibujar)
    confidence: float
    method: str  # 'celdas' | ''
    observation: Observation | None = None


def hypotheses(segs: list[LineSegment], card_height: int) -> list[tuple[list[LineSegment], int]]:
    """(lineas a decodificar, numero de lineas del formato)."""
    out: list[tuple[list[LineSegment], int]] = []
    if len(segs) >= 3:
        out.append((segs[-3:], 3))
        out.append((segs[-2:], 2))
    elif len(segs) == 2:
        out.append((segs, 2))
        step = segs[1].y0 - segs[0].y0
        height = round((segs[0].y1 - segs[0].y0 + segs[1].y1 - segs[1].y0) / 2)
        if 0 < step < height * 4:
            above = LineSegment(
                segs[0].y0 - step, segs[0].y0 - step + height, segs[0].x0, segs[0].x1
            )
            below = LineSegment(
                segs[1].y0 + step, segs[1].y0 + step + height, segs[1].x0, segs[1].x1
            )
            if above.y0 > 0:
                out.append(([above, *segs], 3))
            if below.y1 < card_height:
                out.append(([*segs, below], 3))
    return out


def _score(result: MrzResult | None) -> tuple[int, int]:
    return (
        int(bool(result and result.core_checks_ok)),
        sum(result.checks.values()) if result else 0,
    )


def _expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def read_card(
    engine: Any,
    recognizer: celldecode.LineRecognizer,
    card: np.ndarray,
    variants: tuple[str, ...] = tuple(preprocess.VARIANTS),
    prefer_lines: int | None = None,
    deadline: float | None = None,
) -> CardRead:
    """Segmenta una vez y prueba cada variante de preprocesado hasta que valide un control.

    `deadline` (time.monotonic): pasado ese instante no se empieza trabajo nuevo y se devuelve la
    mejor lectura hasta el momento.
    """
    best: CardRead | None = None
    observation: Observation | None = None
    tried: list[list[LineSegment]] = []
    # la segmentacion depende de un umbral relativo: se prueba sin filas de borde y, si no valida,
    # con ellas (medido: cada modo lee fotos que el otro pierde)
    for drop_border in (True, False):
        if best is not None and _expired(deadline):
            break
        segs = mrzlines.segment_lines(card, drop_border)
        if len(segs) < 2 or segs in tried:
            continue
        tried.append(segs)
        for name in variants:
            if best is not None and _expired(deadline):
                break
            read = _read_variant(
                engine, recognizer, preprocess.VARIANTS[name](card), segs, prefer_lines, deadline
            )
            if read.observation and (
                observation is None or read.observation.confidence > observation.confidence
            ):
                observation = read.observation
            if read.result and read.result.core_checks_ok:
                method = f"{read.method}:{name}"
                return CardRead(read.result, read.segments, read.confidence, method, observation)
            if best is None or (_score(read.result), read.confidence) > (
                _score(best.result),
                best.confidence,
            ):
                best = read
            if name == variants[0] and read.confidence < NO_MRZ_CONFIDENCE:
                break  # nada que se parezca a una MRZ con esta segmentacion
    if best is None:
        return CardRead(None, tuple(mrzlines.segment_lines(card)), 0.0, "")
    return CardRead(best.result, best.segments, best.confidence, best.method, observation)


def _read_variant(
    engine: Any,
    recognizer: celldecode.LineRecognizer,
    card: np.ndarray,
    segs: list[LineSegment],
    prefer_lines: int | None = None,
    deadline: float | None = None,
) -> CardRead:
    best = CardRead(None, tuple(segs), 0.0, "")
    best_key: tuple[int, int, float] = (0, 0, 0.0)
    observation: Observation | None = None
    obs_weight = 0.0
    for index, (lines, count) in enumerate(hypotheses(segs, card.shape[0])):
        if index > 0 and _expired(deadline):
            break
        # borde derecho de las lineas de ESTA hipotesis (no de todas: la pagina de un pasaporte
        # tiene mas texto encima de la MRZ y su mediana puede caer fuera del rango de busqueda)
        x_right = float(np.median([s.x1 for s in lines])) + 6
        x_left = float(np.median([s.x0 for s in lines])) - 2
        decoded, _ = celldecode.decode_lines(
            recognizer,
            card,
            [(s.y0, s.y1) for s in lines],
            celldecode.LAYOUTS[count],
            x_right,
            x_left,
        )
        confidence = float(np.mean([d.confidence for d in decoded]))
        natural = index == 0 or (index == 1 and len(segs) >= 3)  # lineas reales, no inferidas
        weight = PREFERRED_BONUS if count == prefer_lines else 1.0
        if (
            natural
            and (prefer_lines is None or count == prefer_lines)  # no mezclar formatos al fusionar
            and confidence >= MIN_CONFIDENCE
            and (observation is None or confidence * weight > obs_weight)
        ):
            obs_weight = confidence * weight
            # la hipotesis mas confiable (3 lineas = cedula, 2 = pasaporte) alimenta la fusion
            observation = Observation(count, tuple(np.stack(d.scores) for d in decoded), confidence)
        texts = celldecode.snap_fields(
            [d.text for d in decoded], [np.stack(d.scores) for d in decoded], count
        )
        result = parse_mrz(texts)
        key = (*_score(result), confidence)
        if result and confidence >= MIN_CONFIDENCE and key > best_key:
            best, best_key = CardRead(result, tuple(segs), confidence, "celdas"), key
        elif best.result is None and confidence > best.confidence:
            best = CardRead(None, tuple(segs), confidence, "")
        if best.result and best.result.core_checks_ok:
            return CardRead(best.result, best.segments, best.confidence, best.method, observation)
    return CardRead(best.result, best.segments, best.confidence, best.method, observation)
