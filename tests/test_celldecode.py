"""La decodificacion vectorizada por celdas debe dar lo mismo que el bucle original."""

import numpy as np
import pytest

from servicio_mrz import celldecode
from servicio_mrz.celldecode import _CLASSES, ALPHABET, LAYOUTS, _LineProbs


def _reference(lp: _LineProbs, layout: str, x_right: float, pitch: float):
    """Implementacion original (bucle por celda), conservada como referencia."""
    steps = lp.probs.shape[0]
    text, conf, all_scores = [], [], []
    for k in range(len(layout)):
        c0 = x_right - (len(layout) - k) * pitch
        t0 = int(max(0, np.floor((c0 - lp.xa) / lp.px_per_step)))
        t1 = int(min(steps, np.ceil((c0 + pitch - lp.xa) / lp.px_per_step)))
        if t1 <= t0:
            t1 = min(steps, t0 + 1)
        allowed = _CLASSES[layout[k]]
        cols = [ALPHABET.index(c) for c in allowed]
        window = lp.probs[t0:t1][:, cols].max(axis=0)
        scores = window / max(float(window.sum()), 1e-9)
        text.append(allowed[int(scores.argmax())])
        conf.append(float(scores.max()))
        full = np.zeros(len(ALPHABET))
        full[cols] = scores
        all_scores.append(full)
    return "".join(text), float(np.mean(conf)), np.stack(all_scores)


def _line_probs(rng: np.random.Generator, steps: int) -> _LineProbs:
    probs = rng.random((steps, len(ALPHABET))).astype(np.float32) ** 3
    alpha = np.vstack([probs, np.zeros((1, len(ALPHABET)), probs.dtype)])
    return _LineProbs(probs, xa=0, px_per_step=1024 / steps, alpha=alpha)


@pytest.mark.parametrize("lines", [3, 2])
@pytest.mark.parametrize("right", [960.0, 985.0, 1010.0])  # el ultimo se sale por la derecha
@pytest.mark.parametrize("pitch", [19.0, 20.4, 29.5, 31.0])
def test_vectorized_matches_the_original_loop(lines, right, pitch):
    rng = np.random.default_rng(int(right * 10 + pitch))
    for layout in LAYOUTS[lines]:
        lp = _line_probs(rng, steps=256)
        want_text, want_conf, want_scores = _reference(lp, layout, right, pitch)
        got = celldecode._decode_cells(lp, layout, right, pitch)
        assert got.text == want_text
        assert got.confidence == pytest.approx(want_conf, abs=1e-5)
        assert np.allclose(np.stack(got.scores), want_scores, atol=1e-5)


def _peaks(text: str, widths: list[float], start: float, steps: int = 256) -> _LineProbs:
    """Probabilidades de un reconocedor ideal: un pico por caracter, en celdas de ancho variable."""
    probs = np.full((steps, len(ALPHABET)), 0.01, np.float32)
    px_per_step = 1100 / steps
    x = start
    for ch, w in zip(text, widths, strict=True):
        probs[int((x + w / 2) / px_per_step), ALPHABET.index(ch)] = 1.0
        x += w
    alpha = np.vstack([probs, np.zeros((1, len(ALPHABET)), probs.dtype)])
    return _LineProbs(probs, xa=0, px_per_step=px_per_step, alpha=alpha)


TD3_LINE2 = "L898902C<3UTO6908061F9406236ZE184226B<<<<<10"


def test_elastic_grid_recovers_a_line_whose_scale_changes_along_the_page():
    """Pagina curva: la escala crece de forma gradual a lo largo de la linea. La rejilla uniforme
    desalinea las celdas de los extremos; la elastica las sigue."""
    layout = LAYOUTS[2][1]
    widths = [float(w) for w in np.linspace(18.0, 24.0, 44)]
    lp = _peaks(TD3_LINE2, widths, start=60.0)
    right = 60.0 + sum(widths)
    uniform = celldecode._decode_cells(lp, layout, right, pitch=sum(widths) / 44)
    elastic = celldecode._decode_elastic(lp, layout, right, pitch=sum(widths) / 44)
    wrong_elastic = sum(a != b for a, b in zip(elastic.text, TD3_LINE2, strict=True))
    wrong_uniform = sum(a != b for a, b in zip(uniform.text, TD3_LINE2, strict=True))
    # con picos ideales de un solo paso, la ultima celda puede quedar con el pico de su vecina
    assert wrong_elastic <= 1 and wrong_elastic < wrong_uniform


def test_elastic_grid_equals_the_uniform_one_when_the_scale_is_constant():
    layout = LAYOUTS[2][1]
    lp = _peaks(TD3_LINE2, [20.5] * 44, start=70.0)
    assert celldecode._decode_elastic(lp, layout, 70.0 + 44 * 20.5, 20.5).text == TD3_LINE2


def test_peaks_alignment_reads_one_cell_per_emitted_character():
    """Cuando el modelo emite tantos caracteres como celdas, cada pico es una celda, aunque la
    escala cambie a lo largo de la linea y sin ninguna rejilla."""
    layout = LAYOUTS[2][1]
    widths = [float(w) for w in np.linspace(17.0, 25.0, 44)]
    steps = 256
    px_per_step = 1100 / steps
    probs = np.zeros((steps, 1 + len(ALPHABET)), np.float32)  # clase 0 = vacio (CTC)
    probs[:, 0] = 1.0
    x = 60.0
    for ch, w in zip(TD3_LINE2, widths, strict=True):
        t = int((x + w / 2) / px_per_step)
        probs[t, 0], probs[t, 1 + ALPHABET.index(ch)] = 0.0, 1.0
        x += w
    alpha = np.vstack([probs[:, 1:], np.zeros((1, len(ALPHABET)), probs.dtype)])
    lp = _LineProbs(probs, xa=0, px_per_step=px_per_step, alpha=alpha)

    class FakeRecognizer:
        _blank = 0

    got = celldecode._decode_peaks(FakeRecognizer(), lp, layout)
    assert got is not None and got.text == TD3_LINE2
    assert celldecode._decode_peaks(FakeRecognizer(), lp, layout[:-1]) is None  # nº distinto
