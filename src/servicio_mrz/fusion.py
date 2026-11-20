"""Fusion de varios cuadros del MISMO documento: promedia las probabilidades por celda.

Un cuadro suelto puede dejar celdas ambiguas (desenfoque, ruido); los errores de cada cuadro son
parcialmente independientes, asi que el promedio las resuelve. Medido sobre 11 tomas de una misma
cedula: cuadro suelto 36 %, 3 cuadros 64 %, 5 cuadros 80 %, todos 100 %.

Seguridad: la lectura fusionada se acepta SOLO si los digitos de control validan, igual que una
lectura de un cuadro. Nunca se vota a ciegas ni se adivina un dato.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from servicio_mrz import celldecode
from servicio_mrz.celldecode import ALPHABET
from servicio_mrz.mrz import MrzResult, parse_mrz
from servicio_mrz.mrzread import Observation

MAX_FRAMES = 8
MIN_FRAMES = 2
_CLASS_INDEX = {
    cls: [ALPHABET.index(c) for c in chars] for cls, chars in celldecode._CLASSES.items()
}


def _decode(scores: np.ndarray, layout: str) -> str:
    out = []
    for cell in range(scores.shape[0]):
        allowed = celldecode._CLASSES[layout[cell]]
        out.append(allowed[int(np.argmax(scores[cell, _CLASS_INDEX[layout[cell]]]))])
    return "".join(out)


class Fuser:
    """Acumula observaciones del documento actual y propone la lectura fusionada."""

    def __init__(self) -> None:
        self._frames: deque[Observation] = deque(maxlen=MAX_FRAMES)

    def reset(self) -> None:
        self._frames.clear()

    @property
    def size(self) -> int:
        return len(self._frames)

    def add(self, observation: Observation) -> None:
        if self._frames and self._frames[0].lines != observation.lines:
            self._frames.clear()  # otro formato: es otro documento
        self._frames.append(observation)

    def fuse(self) -> MrzResult | None:
        if len(self._frames) < MIN_FRAMES:
            return None
        lines = self._frames[0].lines
        layouts = celldecode.LAYOUTS[lines]
        texts, stacks = [], []
        for i in range(lines):
            stacked = np.mean([f.scores[i] for f in self._frames], axis=0)
            stacks.append(stacked)
            texts.append(_decode(stacked, layouts[i]))
        result = parse_mrz(celldecode.snap_fields(texts, stacks, lines))
        if result and result.core_checks_ok:
            return replace(result, fused_frames=len(self._frames))
        return None
