"""Reconocimiento continuo con camara y lector simulados."""

import threading
import time
from datetime import date

import numpy as np
import pytest

from servicio_mrz import engine
from servicio_mrz.engine import Recognizer
from servicio_mrz.mrz import parse_mrz
from servicio_mrz.ocr import Reading, RowBox

TD3 = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C<3UTO6908061F9406236ZE184226B<<<<<14",
]
EMPTY = np.full((72, 128, 3), 40, np.uint8)
WITH_DOC = np.full((72, 128, 3), 200, np.uint8)


class FakeCamera:
    running = True

    def __init__(self):
        self.frame = EMPTY
        self.n = 0

    def latest(self):
        self.n += 1
        return self.n, self.frame.copy()


class FakeReader:
    """Lee la MRZ solo cuando hay documento; `blind` simula un documento sin MRZ legible."""

    def __init__(self, camera):
        self.camera, self.calls, self.blind = camera, 0, False

    def read_detailed(self, frame):
        self.calls += 1
        has_doc = float(frame.mean()) > 100
        if has_doc and not self.blind:
            result = parse_mrz(TD3, today=date(2026, 11, 20))
            return Reading(result, (RowBox(0.1, 0.5, 0.9, 0.6, 44, True),), boxes=6, ms=5)
        return Reading(None, (), boxes=6 if has_doc else 0, ms=5)


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    monkeypatch.setattr(engine, "POLL_S", 0.01)


def _wait(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def _make(paused=lambda: False):
    cam = FakeCamera()
    reader = FakeReader(cam)
    events: list[dict] = []
    rec = Recognizer(cam, reader, events.append, paused=paused)
    return cam, reader, events, rec


def _kinds(events):
    return [e["tipo"] for e in events]


def test_document_is_read_once_and_json_delivered():
    cam, reader, events, rec = _make()
    rec.start()
    try:
        cam.frame = WITH_DOC  # entra el documento
        assert _wait(lambda: rec.snapshot()["fase"] == "leido")
        time.sleep(0.3)  # mientras siga ahi no debe volver a leer ni a emitir
        calls_after_read = reader.calls
        time.sleep(0.3)
        assert reader.calls == calls_after_read
        assert _kinds(events).count("resultado") == 1
        result = next(e for e in events if e["tipo"] == "resultado")["resultado"]
        assert result["documentType"] == "PASSPORT"
        assert result["fields"]["lastNames"] == "ERIKSSON"
        assert rec.snapshot()["resultado"]["fields"]["documentNumber"] == "L898902C"
    finally:
        rec.stop()


def test_removing_the_document_resets_and_next_document_is_read_again():
    cam, reader, events, rec = _make()
    rec.start()
    try:
        cam.frame = WITH_DOC
        assert _wait(lambda: rec.snapshot()["fase"] == "leido")
        cam.frame = EMPTY  # se retira el documento
        assert _wait(lambda: "retirado" in _kinds(events))
        assert _wait(lambda: rec.snapshot()["fase"] == "buscando")
        assert rec.snapshot()["resultado"] is None
        cam.frame = WITH_DOC  # entra otro
        assert _wait(lambda: _kinds(events).count("resultado") == 2)
    finally:
        rec.stop()


def test_small_scene_changes_do_not_count_as_removal():
    cam, reader, events, rec = _make()
    rec.start()
    try:
        cam.frame = WITH_DOC
        assert _wait(lambda: rec.snapshot()["fase"] == "leido")
        cam.frame = np.clip(WITH_DOC.astype(int) + 5, 0, 255).astype(np.uint8)  # ruido/luz leve
        time.sleep(0.5)
        assert rec.snapshot()["fase"] == "leido" and "retirado" not in _kinds(events)
    finally:
        rec.stop()


def test_text_without_mrz_suggests_turning_the_document(monkeypatch):
    monkeypatch.setattr(engine, "IDLE_RETRY_S", 0.01)  # sin tarjeta, reintenta con el latido
    cam, reader, events, rec = _make()
    reader.blind = True
    rec.start()
    try:
        cam.frame = WITH_DOC  # hay texto (6 cajas) pero ninguna fila MRZ
        assert _wait(lambda: "sin_mrz" in _kinds(events))
        assert "otra cara" in rec.snapshot()["pista"]
        assert rec.snapshot()["fase"] == "buscando"
    finally:
        rec.stop()


def test_empty_scene_does_not_burn_cpu_and_never_suggests_turning(monkeypatch):
    monkeypatch.setattr(engine, "IDLE_RETRY_S", 60.0)  # sin "latido": solo reacciona a cambios
    cam, reader, events, rec = _make()
    rec.start()
    try:
        assert _wait(lambda: rec.snapshot()["intentos"] >= 1)
        time.sleep(0.5)
        assert rec.snapshot()["intentos"] == 1  # escena vacia e inmovil: no insiste
        assert "sin_mrz" not in _kinds(events) and rec.snapshot()["pista"] is None
        cam.frame = WITH_DOC  # entra algo: reacciona de inmediato
        assert _wait(lambda: rec.snapshot()["fase"] == "leido")
    finally:
        rec.stop()


def test_idle_scene_is_retried_periodically_in_case_document_blends_with_background(monkeypatch):
    monkeypatch.setattr(engine, "IDLE_RETRY_S", 0.2)
    cam, reader, events, rec = _make()
    rec.start()
    try:
        assert _wait(lambda: rec.snapshot()["intentos"] >= 3)
    finally:
        rec.stop()


def test_paused_while_calibrating_and_resumes():
    flag = threading.Event()
    cam, reader, events, rec = _make(paused=flag.is_set)
    flag.set()
    rec.start()
    try:
        cam.frame = WITH_DOC
        assert _wait(lambda: rec.snapshot()["fase"] == "pausado")
        time.sleep(0.2)
        assert reader.calls == 0
        flag.clear()
        assert _wait(lambda: rec.snapshot()["fase"] == "leido")
    finally:
        rec.stop()


def test_websocket_sends_current_state_on_connect():
    from fastapi.testclient import TestClient

    from servicio_mrz.app import create_app
    from servicio_mrz.config import Settings

    app = create_app(Settings(camera_enabled=False, _env_file=None))
    with TestClient(app) as client, client.websocket_connect("/eventos") as ws:
        first = ws.receive_json()
        assert first["tipo"] == "estado" and first["fase"] == "buscando"
        assert client.get("/estado").json()["fase"] == "buscando"


def test_card_without_mrz_suggests_turning_quickly_without_idling():
    """Tarjeta localizada (el modelo la ve) pero sin lineas MRZ: es la cara sin MRZ."""
    cam = FakeCamera()
    events: list[dict] = []
    square = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))

    class FrontSideReader:
        def read_detailed(self, frame):
            card = square if float(frame.mean()) > 100 else None
            return Reading(None, (), boxes=0, ms=5, card=card)

    rec = Recognizer(cam, FrontSideReader(), events.append)
    rec.start()
    try:
        cam.frame = WITH_DOC
        assert _wait(lambda: "sin_mrz" in _kinds(events))
        state = rec.snapshot()
        assert state["tarjeta"] is not None and len(state["tarjeta"]) == 4
        assert "otra cara" in state["pista"] and state["intentos"] >= 3
    finally:
        rec.stop()
