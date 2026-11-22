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


def test_a_sideways_card_skips_the_mrz_stage(monkeypatch):
    """Texto vertical: la MRZ no se lee en 0/180 grados, no se gastan segundos intentandolo."""
    from types import SimpleNamespace

    from servicio_mrz import ocr

    located = SimpleNamespace(corners=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)), aspect=1.5)
    warped = SimpleNamespace(image=np.zeros((680, 1024, 3), np.uint8), located=located)

    def forbidden(*a, **k):
        raise AssertionError("no debe leer MRZ en una tarjeta de lado")

    monkeypatch.setattr(ocr.cardfinder, "locate_card", lambda img: located)
    monkeypatch.setattr(ocr.cardfinder, "warp_card", lambda img, loc: warped)
    monkeypatch.setattr(ocr.orientation, "assess", lambda engine, card: (None, 0.0))
    monkeypatch.setattr(ocr.orientation, "sideways", lambda engine, card: True)
    monkeypatch.setattr(ocr.mrzread, "read_card", forbidden)
    monkeypatch.setattr(ocr.celldecode, "LineRecognizer", lambda engine: object())
    reader = ocr.MrzReader()
    monkeypatch.setattr(reader, "_engine_ready", lambda: object())
    reading = reader.read_detailed(np.zeros((1080, 1920, 3), np.uint8))
    assert reading.result is None and reading.card is not None and reading.card_aspect == 1.5
