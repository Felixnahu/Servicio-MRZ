"""Decodificacion de lineas MRZ celda por celda, con alfabeto restringido por posicion.

La MRZ es monoespaciada (30 o 44 celdas por linea) y el estandar ICAO 9303 fija que tipo de
caracter admite cada posicion (letras, digitos, o ambos). El reconocedor general tiende a
interpretar toda una linea como "numero" o como "texto" y fuerza los caracteres (I->1, B->8,
'<'->otro digito). Aqui se usan sus PROBABILIDADES por paso de tiempo y, en cada celda, se elige el
mejor caracter PERMITIDO en esa posicion. Los digitos de control siguen siendo el arbitro final.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALPHABET = DIGITS + LETTERS + "<"

_DIG, _LET, _LETF, _ANY, _SEX = "d", "l", "f", "a", "s"  # f = letra o '<'
_CLASSES = {
    _DIG: DIGITS,
    _LET: LETTERS,
    _LETF: LETTERS + "<",
    _ANY: ALPHABET,
    _SEX: "MF<",
}


def _layout(spec: list[tuple[int, str]]) -> str:
    return "".join(cls * n for n, cls in spec)


# ICAO 9303: clase de caracter permitida en cada posicion
TD1_LAYOUT = [
    _layout([(2, _LETF), (3, _LET), (9, _ANY), (1, _DIG), (15, _ANY)]),
    _layout([(7, _DIG), (1, _SEX), (7, _DIG), (3, _LET), (11, _ANY), (1, _DIG)]),
    _layout([(30, _LETF)]),
]
TD3_LAYOUT = [
    _layout([(2, _LETF), (3, _LETF), (39, _LETF)]),
    _layout(
        [(9, _ANY), (1, _DIG), (3, _LET), (7, _DIG), (1, _SEX), (7, _DIG), (14, _ANY), (2, _DIG)]
    ),
]
LAYOUTS = {3: TD1_LAYOUT, 2: TD3_LAYOUT}


@dataclass(frozen=True)
class Decoded:
    text: str
    confidence: float  # promedio de la probabilidad normalizada de la celda elegida
    scores: tuple[np.ndarray, ...] = ()  # por celda: probabilidad de cada caracter de ALPHABET


class LineRecognizer:
    """Acceso a las probabilidades por paso de tiempo del reconocedor de RapidOCR."""

    def __init__(self, engine: Any) -> None:
        rec = engine.text_rec
        self._rec = rec
        charlist = rec.postprocess_op.character
        self._index = {c: charlist.index(c) for c in ALPHABET}
        self._blank = 0

    def probabilities(self, crop_bgr: np.ndarray) -> np.ndarray:
        """(T, clases) del modelo para un recorte de una linea ya redimensionado a alto 48."""
        ratio = max(crop_bgr.shape[1] / crop_bgr.shape[0], 320 / 48)
        x = self._rec.resize_norm_img(crop_bgr, ratio)
        out = self._rec.session(x[None].astype(np.float32))
        return (out[0] if isinstance(out, (list, tuple)) else out)[0]

    def peak_positions(self, probs: np.ndarray) -> np.ndarray:
        """Pasos de tiempo donde el modelo 'dispara' un caracter (no vacio, con confianza)."""
        best = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        return np.nonzero((best != self._blank) & (conf > 0.3))[0]

    def column(self, char: str) -> int:
        """Columna del modelo que corresponde a un caracter de ALPHABET."""
        return self._index[char]

    def restrict(self, probs: np.ndarray, allowed: str) -> tuple[np.ndarray, str]:
        cols = [self._index[c] for c in allowed]
        return probs[:, cols], allowed


# Paso de caracter en la tarjeta enderezada (1024 px de ancho): es una constante fisica del estandar
# (10 caracteres por pulgada). Medido en cedulas bolivianas TD1: 29.5 px; TD3 (125 mm) se deduce.
PITCH_TD1 = 29.5
PITCH_TD3 = PITCH_TD1 * 85.6 / 125.0
PITCH_TOLERANCE = 0.05
PITCH_STEP = 0.5
REFINE_ROUNDS = 2  # rondas de afinado alterno desplazamiento/paso (ver decode_lines)
SHIFT_RANGE = 24  # px de busqueda del borde derecho alrededor de la estimacion inicial
MAX_PITCH = PITCH_TD1 * (1 + PITCH_TOLERANCE)


@dataclass(frozen=True)
class _LineProbs:
    probs: np.ndarray  # (T, clases)
    xa: int  # borde izquierdo del recorte, en px de tarjeta
    px_per_step: float
    alpha: np.ndarray  # (T + 1, len(ALPHABET)): probabilidades por caracter + una fila de ceros


def _line_probs(
    recognizer: LineRecognizer, card: np.ndarray, y0: int, y1: int, cells: int, x_right: float
) -> _LineProbs:
    pad_y = int((y1 - y0) * 0.35)
    xa = max(0, int(x_right - cells * MAX_PITCH) - 20)
    xb = min(card.shape[1], int(x_right) + 24)
    crop = card[max(0, y0 - pad_y) : y1 + pad_y, xa:xb]
    crop = cv2.resize(
        crop, None, fx=48 / crop.shape[0], fy=48 / crop.shape[0], interpolation=cv2.INTER_CUBIC
    )
    probs = recognizer.probabilities(crop)
    columns = [recognizer.column(c) for c in ALPHABET]
    alpha = np.vstack([probs[:, columns], np.zeros((1, len(ALPHABET)), probs.dtype)])
    return _LineProbs(probs, xa, (xb - xa) / probs.shape[0], alpha)


_MASKS: dict[str, np.ndarray] = {}


def _mask(layout: str) -> np.ndarray:
    """(celdas, len(ALPHABET)): 1 donde el caracter esta permitido en esa posicion."""
    mask = _MASKS.get(layout)
    if mask is None:
        mask = np.zeros((len(layout), len(ALPHABET)), np.float32)
        for k, cls in enumerate(layout):
            mask[k, [ALPHABET.index(c) for c in _CLASSES[cls]]] = 1.0
        _MASKS[layout] = mask
    return mask


def _cell_scores(lp: _LineProbs, layout: str, x_right: float, pitch: float) -> np.ndarray:
    """(celdas, len(ALPHABET)): puntuacion normalizada de cada caracter permitido, por celda.

    Cada celda toma el maximo de las probabilidades del reconocedor en su ventana de tiempo y se
    restringe al alfabeto de su posicion. Vectorizado: se evalua ~70 veces por lectura.
    """
    steps = lp.probs.shape[0]
    cells = len(layout)
    c0 = x_right - (cells - np.arange(cells)) * pitch
    t0 = np.minimum(np.maximum(np.floor((c0 - lp.xa) / lp.px_per_step), 0), steps - 1).astype(int)
    t1 = np.minimum(np.ceil((c0 + pitch - lp.xa) / lp.px_per_step), steps).astype(int)
    t1 = np.where(t1 <= t0, np.minimum(steps, t0 + 1), t1)
    bounds = np.empty(2 * cells, dtype=int)
    bounds[0::2], bounds[1::2] = t0, t1
    window = np.maximum.reduceat(lp.alpha, bounds, axis=0)[0::2] * _mask(layout)
    return window / np.maximum(window.sum(axis=1, keepdims=True), 1e-9)


def _decode_cells(lp: _LineProbs, layout: str, x_right: float, pitch: float) -> Decoded:
    scores = _cell_scores(lp, layout, x_right, pitch)
    best = scores.argmax(axis=1)
    text = "".join(ALPHABET[i] for i in best)
    confidences = scores[np.arange(len(best)), best]
    return Decoded(text, float(confidences.mean()), tuple(scores.astype(np.float64)))


def decode_lines(
    recognizer: LineRecognizer,
    card: np.ndarray,
    segments: list[tuple[int, int]],
    layouts: list[str],
    x_right: float,
    x_left: float | None = None,
    with_elastic: bool = False,
) -> tuple[list[Decoded], float, list[list[Decoded]]]:
    """Decodifica las lineas (y0, y1) con la rejilla que mejor explica al reconocedor.

    Devuelve (lineas, paso, alternativas). Con `with_elastic` las alternativas son la misma lectura
    con otras alineaciones, que solo se usan si la uniforme no valida los controles:
    1. por picos del reconocedor: cada caracter que el modelo emite es una celda (si el numero de
       caracteres emitidos coincide con el de celdas); no depende de ninguna rejilla;
    2. rejilla elastica: cada celda puede variar un poco de ancho (pagina curva de un pasaporte),
       alineada por programacion dinamica y anclada a la rejilla uniforme.

    El paso de caracter y el borde derecho son comunes a todas las lineas. Busqueda en dos fases:
    primero el desplazamiento horizontal con el paso nominal, luego el paso con ese desplazamiento.
    (Se probo un afinado fino adicional: no mejoro la lectura y costaba ~2x tiempo.)
    """
    nominal = PITCH_TD1 if len(layouts) == 3 else PITCH_TD3
    # Anclajes del borde derecho de la rejilla: el que da la tinta de la linea y el deducido del
    # borde izquierdo (los '<' finales tienen poco contraste y la tinta puede acabar ~100 px
    # corta en fotos oscuras). El recorte que ve el modelo debe cubrir el mas a la derecha.
    anchors = [x_right]
    if x_left is not None:
        anchors.append(x_left + len(layouts[0]) * nominal)
    line_probs = [
        _line_probs(recognizer, card, y0, y1, len(layouts[i]), max(anchors))
        for i, (y0, y1) in enumerate(segments)
    ]

    def mean_confidence(right: float, pitch: float) -> float:
        return float(
            np.mean(
                [
                    _cell_scores(lp, layouts[i], right, pitch).max(axis=1).mean()
                    for i, lp in enumerate(line_probs)
                ]
            )
        )

    right = max(
        (a + dx for a in anchors for dx in range(-SHIFT_RANGE, SHIFT_RANGE + 1, 2)),
        key=lambda r: mean_confidence(r, nominal),
    )
    low, high = nominal * (1 - PITCH_TOLERANCE), nominal * (1 + PITCH_TOLERANCE)
    pitch = float(
        max(np.arange(low, high + 1e-6, PITCH_STEP), key=lambda q: mean_confidence(right, float(q)))
    )
    # El desplazamiento de la 1a fase se eligio con el paso nominal; si el paso real difiere (p. ej.
    # 2 % en TD3, 44 celdas) queda sesgado y el error se acumula en el borde izquierdo. Se alterna.
    for _ in range(REFINE_ROUNDS):
        right = max(
            (right + dx for dx in np.arange(-6, 6.01, 1.5)), key=lambda r: mean_confidence(r, pitch)
        )
        pitch = max(
            (pitch * (1 + f) for f in np.arange(-0.02, 0.0201, 0.005)),
            key=lambda q: mean_confidence(right, float(q)),
        )
    uniform = [
        _decode_cells(lp, layouts[i], right, float(pitch)) for i, lp in enumerate(line_probs)
    ]
    alternatives: list[list[Decoded]] = []
    if with_elastic:
        by_peaks = [
            _decode_peaks(recognizer, lp, layouts[i]) or uniform[i]
            for i, lp in enumerate(line_probs)
        ]
        alternatives.append(by_peaks)
        alternatives.append(
            [
                _decode_elastic(lp, layouts[i], right, float(pitch))
                for i, lp in enumerate(line_probs)
            ]
        )
    return uniform, float(pitch), alternatives


def _decode_peaks(recognizer: LineRecognizer, lp: _LineProbs, layout: str) -> Decoded | None:
    """Una celda por caracter emitido por el modelo (None si no hay tantos como celdas)."""
    best = lp.probs.argmax(axis=1)
    runs: list[tuple[int, int]] = []  # [inicio, fin) de cada emision no vacia
    start = None
    for t in range(len(best) + 1):
        cur = int(best[t]) if t < len(best) else 0
        prev = int(best[t - 1]) if t > 0 else 0
        if start is not None and (cur != prev or cur == recognizer._blank):
            runs.append((start, t))
            start = None
        if start is None and cur != recognizer._blank and t < len(best):
            start = t
    if len(runs) != len(layout):
        return None
    mask = _mask(layout)
    scores = np.zeros((len(layout), len(ALPHABET)), np.float32)
    for c, (a, b) in enumerate(runs):
        window = lp.alpha[max(0, a - 1) : min(len(lp.probs), b + 1)].max(axis=0) * mask[c]
        scores[c] = window / max(float(window.sum()), 1e-9)
    chosen = scores.argmax(axis=1)
    text = "".join(ALPHABET[i] for i in chosen)
    confidences = scores[np.arange(len(layout)), chosen]
    return Decoded(text, float(confidences.mean()), tuple(scores.astype(np.float64)))


# --- rejilla elastica ---------------------------------------------------------------------------

ELASTIC_UPSAMPLE = 4  # subdivisiones por paso del reconocedor (resolucion de las fronteras)
ELASTIC_WIDTH_TOL = 0.2  # cada celda puede medir +-20 % del paso uniforme
ELASTIC_PENALTY = 1.0  # castigo (en puntos de confianza) por (desviacion relativa)^2
ELASTIC_EDGE_CELLS = 1.5  # cuanto pueden moverse los extremos de la linea (en celdas)


def _decode_elastic(lp: _LineProbs, layout: str, x_right: float, pitch: float) -> Decoded:
    """Alinea las celdas de la linea con anchos variables (DP) y decodifica cada una."""
    cells = len(layout)
    k = ELASTIC_UPSAMPLE
    fine = np.repeat(lp.alpha[:-1], k, axis=0)  # (N, A)
    n = fine.shape[0]
    unit = lp.px_per_step / k  # px por unidad fina
    w0 = pitch / unit
    lmin = max(1, int(np.floor(w0 * (1 - ELASTIC_WIDTH_TOL))))
    lmax = int(np.ceil(w0 * (1 + ELASTIC_WIDTH_TOL)))
    widths = range(lmin, lmax + 1)
    # ventanas de cada ancho: windows[l][s] = maximo de fine[s : s + l]
    windows: dict[int, np.ndarray] = {}
    cur = fine
    for width in range(2, lmax + 1):
        cur = np.maximum(cur[: n - width + 1], fine[width - 1 :])
        windows[width] = cur
    windows[1] = fine
    mask = _mask(layout)
    start0 = (x_right - cells * pitch - lp.xa) / unit
    end0 = (x_right - lp.xa) / unit
    slack = ELASTIC_EDGE_CELLS * w0
    neg = -1e9
    prev = np.full(n + 1, neg)
    lo, hi = int(max(0, start0 - slack)), int(min(n, start0 + slack))
    prev[lo : hi + 1] = 0.0
    choice = np.zeros((cells, n + 1), dtype=np.int16)
    for c in range(cells):
        new = np.full(n + 1, neg)
        pick = np.zeros(n + 1, dtype=np.int16)
        for width in widths:
            win = windows.get(width)
            if win is None or win.shape[0] == 0:
                continue
            masked = win * mask[c]
            score = masked.max(axis=1) / np.maximum(masked.sum(axis=1), 1e-9)
            penalty = ELASTIC_PENALTY * ((width - w0) / w0) ** 2
            cand = prev[: win.shape[0]] + score - penalty
            target = slice(width, width + win.shape[0])
            better = cand > new[target]
            new[target] = np.where(better, cand, new[target])
            pick[target] = np.where(better, width, pick[target])
        prev, choice[c] = new, pick
    lo_e, hi_e = int(max(0, end0 - slack)), int(min(n, end0 + slack))
    if hi_e < lo_e or lo > hi:  # la linea cae fuera del recorte: sin alineacion elastica posible
        return _decode_cells(lp, layout, x_right, pitch)
    end = lo_e + int(np.argmax(prev[lo_e : hi_e + 1]))
    if prev[end] < -1e8:
        return _decode_cells(lp, layout, x_right, pitch)
    # retroceso: ancho elegido de cada celda, de la ultima a la primera
    bounds = [end]
    for c in range(cells - 1, -1, -1):
        step = int(choice[c, bounds[-1]])
        if step <= 0:
            return _decode_cells(lp, layout, x_right, pitch)
        bounds.append(bounds[-1] - step)
    bounds.reverse()
    scores = np.zeros((cells, len(ALPHABET)), np.float32)
    for c in range(cells):
        s_, e_ = bounds[c], bounds[c + 1]
        window = fine[s_:e_].max(axis=0) * mask[c] if e_ > s_ else mask[c] * 0
        scores[c] = window / max(float(window.sum()), 1e-9)
    best = scores.argmax(axis=1)
    text = "".join(ALPHABET[i] for i in best)
    confidences = scores[np.arange(cells), best]
    return Decoded(text, float(confidences.mean()), tuple(scores.astype(np.float64)))


# --- campos de dominio cerrado (no cubiertos por digitos de control) ---------------------------

# formato (numero de lineas) -> (linea y rango del emisor, linea y rango de la nacionalidad,
# letras posibles del tipo de documento)
_CLOSED_FIELDS = {
    3: ((0, 2, 5), (1, 15, 18), "IAC"),  # TD1: cedulas
    2: ((0, 2, 5), (1, 10, 13), "PV"),  # TD3: pasaportes y visados
}
_EPS = 1e-6
_EQUAL_MARGIN = 3.0  # nats: cuanto peor puede puntuar "emisor = nacionalidad" y aun preferirse
_FILLER_PRIOR = 3.0  # el 2o caracter del tipo es casi siempre '<' (p. ej. "I<", "P<")


def snap_fields(texts: list[str], scores: list[np.ndarray], lines: int) -> list[str]:
    """Corrige tipo de documento, pais emisor y nacionalidad al valor VALIDO mas probable.

    Los digitos de control no cubren estos campos: el OCR puede acertar numero y fechas y fallar
    en 'BOL' o leer 'AA' en un pasaporte (la app decide el tipo con ellos). Son de dominio
    cerrado, asi que en vez de descartar la lectura se elige, con las probabilidades de cada
    celda, el valor valido mejor puntuado: tipo segun el formato (TD3 = P/V, TD1 = I/A/C), codigos
    de pais reales, y emisor = nacionalidad salvo que las celdas lo contradigan con claridad.
    El resultado sigue sujeto a los digitos de control y a `MrzResult.plausible`.
    """
    from servicio_mrz.mrz.countries import VALID_CODES

    spec = _CLOSED_FIELDS.get(lines)
    if spec is None or len(texts) < 2:
        return texts
    (l_iss, a_iss, b_iss), (l_nat, a_nat, b_nat), doc_first = spec
    if len(texts[l_iss]) < b_iss or len(texts[l_nat]) < b_nat:
        return texts
    codes = sorted(VALID_CODES)
    idx = np.array([[ALPHABET.index(ch) for ch in c] for c in codes])  # (N, 3)

    def logp(line: int, a: int, b: int) -> np.ndarray:
        cells = np.log(scores[line][a:b] + _EPS)
        return sum(cells[j][idx[:, j]] for j in range(3))  # (N,) por codigo candidato

    iss, nat = logp(l_iss, a_iss, b_iss), logp(l_nat, a_nat, b_nat)
    best_iss, best_nat = int(np.argmax(iss)), int(np.argmax(nat))
    together = iss + nat
    equal = int(np.argmax(together))  # mejor codigo comun
    if together[equal] >= iss[best_iss] + nat[best_nat] - _EQUAL_MARGIN:
        best_iss = best_nat = equal
    out = list(texts)
    out[l_iss] = out[l_iss][:a_iss] + codes[best_iss] + out[l_iss][b_iss:]
    out[l_nat] = out[l_nat][:a_nat] + codes[best_nat] + out[l_nat][b_nat:]

    first = scores[0][0:2]
    letter = max(doc_first, key=lambda ch: first[0][ALPHABET.index(ch)])
    filler = np.log(first[1] + _EPS)
    filler[ALPHABET.index("<")] += _FILLER_PRIOR
    second = ALPHABET[int(np.argmax(filler))]
    out[0] = letter + second + out[0][2:]
    return out
