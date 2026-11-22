"""Lee la MRZ de una tarjeta ya enderezada: segmenta lineas, formula hipotesis y decide con los
digitos de control.

Hipotesis (la segmentacion puede perder una linea o la tarjeta ser un pasaporte):
  - 3 lineas halladas        -> TD1 (cedula)
  - 2 lineas halladas        -> TD3 (pasaporte, 44 celdas) o TD1 con una linea perdida
                                 (la linea faltante se prueba arriba y abajo, por espaciado)
Cada hipotesis se decodifica celda por celda y se parsea; gana la que valida los controles.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from servicio_mrz import celldecode, mrzlines, preprocess
from servicio_mrz.mrz import MrzResult, parse_mrz
from servicio_mrz.mrzlines import LineSegment

# Una MRZ legible da >= 0.84 con la 1a variante y el texto normal <= 0.47 (medido). Por debajo de
# este umbral no se prueban mas variantes: no hay MRZ legible y el OCR visual toma el relevo.
NO_MRZ_CONFIDENCE = 0.55  # medido: bajo esto la 1a variante lee basura y las demas tambien
# Correccion de una cifra con los digitos de control (ver repair_checks): solo se prueban las celdas
# donde el reconocedor dudo de verdad (la 2.a opcion llega al 50 % de la elegida y a 12 % absoluto).
REPAIR_MIN_RATIO = 0.5
REPAIR_MIN_P = 0.12
REPAIR_MAX_CANDIDATES = 12
PREFERRED_BONUS = 1.2  # a igual lectura, se prefiere el formato que sugiere la forma de la tarjeta
EDGE_STRIP_RATIO = 0.7  # una banda final mas fina que esto (vs la mediana) es borde, no linea
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


def _without_edge_strip(segs: list[LineSegment]) -> list[LineSegment]:
    """Quita la ultima banda si es mucho mas delgada que las demas: es el borde inferior de la
    tarjeta (el recorte puede incluir algo de la mesa), no una linea de la MRZ. Sin esto, la
    hipotesis "las 3 ultimas lineas" toma una linea de la MRZ de menos y la franja del borde."""
    if len(segs) < 4:
        return segs
    heights = sorted(s.y1 - s.y0 for s in segs[:-1])
    median = heights[len(heights) // 2]
    last = segs[-1].y1 - segs[-1].y0
    return segs[:-1] if last < EDGE_STRIP_RATIO * median else segs


def hypotheses(segs: list[LineSegment], card_height: int) -> list[tuple[list[LineSegment], int]]:
    """(lineas a decodificar, numero de lineas del formato)."""
    out: list[tuple[list[LineSegment], int]] = []
    segs = _without_edge_strip(segs)
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


def repair_checks(texts: list[str], scores: list[np.ndarray]) -> MrzResult | None:
    """Corrige UNA cifra mal leida cuando los digitos de control no cuadran.

    El OCR a veces duda entre dos cifras parecidas (6/9, 8/0...) y elige la equivocada: el numero
    de documento deja de validar aunque todo lo demas este bien (caso real: cedula 2023 con
    confianza 0.96 no leida porque 6 se leyo como 9, con 0.34 contra 0.27). Se prueba, de la celda
    mas dudosa a la menos, sustituirla por su 2.a o 3.a opcion; solo se acepta si TODOS los
    controles (numero, nacimiento, vencimiento y el compuesto) validan y la forma es plausible.
    Con el compuesto de por medio una sustitucion equivocada pasa por azar ~1 de cada 100 y solo
    se prueban las celdas dudosas (2-5 por foto), asi que el riesgo es del orden del 1 %.
    """
    base = parse_mrz(texts)
    if base is None or base.core_checks_ok:
        return None
    alphabet = celldecode.ALPHABET
    options: list[tuple[float, int, int, str]] = []
    for line, (text, probs) in enumerate(zip(texts, scores, strict=True)):
        for cell, char in enumerate(text):
            row = probs[cell]
            chosen = float(row[alphabet.index(char)])
            for index in np.argsort(-row)[:3]:
                alt = alphabet[int(index)]
                p = float(row[index])
                if alt != char and p >= max(REPAIR_MIN_P, REPAIR_MIN_RATIO * chosen):
                    cost = math.log(p + 1e-9) - math.log(chosen + 1e-9)
                    options.append((cost, line, cell, alt))
    options.sort(reverse=True)  # menor penalizacion primero
    for _, line, cell, alt in options[:REPAIR_MAX_CANDIDATES]:
        fixed = list(texts)
        fixed[line] = fixed[line][:cell] + alt + fixed[line][cell + 1 :]
        result = parse_mrz(fixed)
        if result and result.core_checks_ok and result.checks.get("composite", False):
            return replace(result, repaired=1)
    return None


def _parse(texts: list[str], scores: list[np.ndarray]) -> MrzResult | None:
    """parse_mrz y, si no valida por una cifra dudosa, esa misma lectura con la cifra corregida."""
    result = parse_mrz(texts)
    if result is not None and not result.core_checks_ok:
        result = repair_checks(texts, scores) or result
    return result


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
        decoded, _, alternatives = celldecode.decode_lines(
            recognizer,
            card,
            [(s.y0, s.y1) for s in lines],
            celldecode.LAYOUTS[count],
            x_right,
            x_left,
            with_elastic=True,
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
        result = _parse(texts, [np.stack(d.scores) for d in decoded])
        for alt in alternatives:
            if result and result.core_checks_ok:
                break
            # rejilla uniforme sin validar: probar las alineaciones alternativas (picos del modelo,
            # rejilla elastica) y quedarse con una solo si valida los controles
            texts_alt = celldecode.snap_fields(
                [d.text for d in alt], [np.stack(d.scores) for d in alt], count
            )
            result_alt = _parse(texts_alt, [np.stack(d.scores) for d in alt])
            if result_alt and result_alt.core_checks_ok:
                decoded, result = alt, result_alt
                confidence = float(np.mean([d.confidence for d in decoded]))
        key = (*_score(result), confidence)
        if result and confidence >= MIN_CONFIDENCE and key > best_key:
            best, best_key = CardRead(result, tuple(segs), confidence, "celdas"), key
        elif best.result is None and confidence > best.confidence:
            best = CardRead(None, tuple(segs), confidence, "")
        if best.result and best.result.core_checks_ok:
            return CardRead(best.result, best.segments, best.confidence, best.method, observation)
    return CardRead(best.result, best.segments, best.confidence, best.method, observation)
