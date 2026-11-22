"""Sentido de la tarjeta. Motor falso: cajas y etiquetas sinteticas, sin fotos."""

import numpy as np

from servicio_mrz import orientation


class FakeEngine:
    def __init__(self, labels, boxes=None):
        self._labels = labels
        self._boxes = boxes if boxes is not None else [self._line(i) for i in range(len(labels))]

    @staticmethod
    def _line(i, width=200.0, height=30.0):
        y = 10.0 + 40 * i
        return np.array([[0, y], [width, y], [width, y + height], [0, y + height]], float)

    def text_det(self, card):
        return (np.array(self._boxes) if self._boxes else None), 0.0

    def get_crop_img_list(self, card, boxes):
        return [None] * len(boxes)

    def text_cls(self, crops):
        return crops, [[label, 0.99] for label in self._labels[: len(crops)]], 0.0


CARD = np.zeros((300, 400, 3), np.uint8)


def test_majority_of_upright_lines_is_not_upside_down():
    assert orientation.upside_down(FakeEngine(["0", "0", "0", "180"]), CARD) is False


def test_majority_of_reversed_lines_is_upside_down():
    assert orientation.upside_down(FakeEngine(["180", "180", "180", "0"]), CARD) is True


def test_too_few_lines_gives_no_answer():
    assert orientation.upside_down(FakeEngine(["180", "180"]), CARD) is None


def test_a_tie_gives_no_answer():
    assert orientation.upside_down(FakeEngine(["0", "180", "0", "180"]), CARD) is None


def test_vertical_or_tiny_boxes_do_not_vote():
    vertical = [FakeEngine._line(i, width=20, height=120) for i in range(5)]
    assert orientation.upside_down(FakeEngine(["180"] * 5, vertical), CARD) is None


def test_no_text_gives_no_answer():
    assert orientation.upside_down(FakeEngine([], boxes=[]), CARD) is None


def test_an_engine_failure_never_raises():
    class Broken:
        def text_det(self, card):
            raise RuntimeError("modelo cambiado")

    assert orientation.upside_down(Broken(), CARD) is None


def test_mostly_vertical_text_means_sideways():
    vertical = [FakeEngine._line(i, width=30, height=150) for i in range(4)]
    assert orientation.sideways(FakeEngine(["0"] * 4, vertical), CARD) is True


def test_horizontal_text_is_not_sideways():
    assert orientation.sideways(FakeEngine(["0", "0", "0", "0"]), CARD) is False
    assert orientation.sideways(FakeEngine([], boxes=[]), CARD) is False
