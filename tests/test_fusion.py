"""Fusion de cuadros: promedia probabilidades por celda y solo acepta lo que valida."""

import numpy as np

from servicio_mrz import celldecode
from servicio_mrz.celldecode import ALPHABET
from servicio_mrz.fusion import MIN_FRAMES, Fuser
from servicio_mrz.mrzread import Observation

TD3 = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C<3UTO6908061F9406236ZE184226B<<<<<14",
]
LAYOUT = celldecode.LAYOUTS[2]


def _scores(lines: list[str], wrong: dict[tuple[int, int], tuple[str, float]], rng) -> tuple:
    """Probabilidades sinteticas: el caracter verdadero con 0.8, ruido repartido; `wrong` fuerza
    celdas donde este cuadro 'cree' otro caracter con la probabilidad dada."""
    out = []
    for i, line in enumerate(lines):
        cells = np.full((len(line), len(ALPHABET)), 0.2 / (len(ALPHABET) - 1))
        for k, ch in enumerate(line):
            cells[k, ALPHABET.index(ch)] = 0.8
            if (i, k) in wrong:
                bad, p = wrong[(i, k)]
                cells[k] = 0.2 / (len(ALPHABET) - 2)
                cells[k, ALPHABET.index(bad)] = p
                cells[k, ALPHABET.index(ch)] = 1.0 - p - 0.2
            cells[k] += rng.uniform(0, 0.01, len(ALPHABET))
            cells[k] /= cells[k].sum()
        out.append(cells)
    return tuple(out)


def _obs(wrong):
    return Observation(2, _scores(TD3, wrong, np.random.default_rng(len(wrong))), 0.6)


def test_not_enough_frames_gives_nothing():
    fuser = Fuser()
    fuser.add(_obs({}))
    assert MIN_FRAMES == 2 and fuser.fuse() is None


def test_errors_that_differ_between_frames_are_resolved_by_averaging():
    fuser = Fuser()
    # cada cuadro se equivoca en OTRA celda del numero de documento (linea 2, celdas 0..8)
    fuser.add(_obs({(1, 1): ("0", 0.55)}))
    fuser.add(_obs({(1, 4): ("5", 0.55)}))
    fuser.add(_obs({(1, 6): ("1", 0.55)}))
    result = fuser.fuse()
    assert result is not None and result.core_checks_ok
    assert result.document_number == "L898902C" and result.fused_frames == 3


def test_a_consistent_error_is_never_accepted_because_check_digits_fail():
    fuser = Fuser()
    for _ in range(4):  # todos los cuadros leen mal la MISMA celda: la fusion no puede arreglarlo
        fuser.add(_obs({(1, 3): ("9", 0.7)}))
    assert fuser.fuse() is None  # el digito de control lo detecta: no se entrega un dato falso


def test_reset_and_format_change_clear_the_buffer():
    fuser = Fuser()
    fuser.add(_obs({}))
    fuser.reset()
    assert fuser.size == 0
    fuser.add(_obs({}))
    other = Observation(
        3, tuple(np.ones((30, len(ALPHABET))) / len(ALPHABET) for _ in range(3)), 0.6
    )
    fuser.add(other)
    assert fuser.size == 1  # otro formato = otro documento
