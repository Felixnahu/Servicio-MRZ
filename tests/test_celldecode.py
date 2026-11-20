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
