"""Regresion: Ctrl+C no terminaba el proceso si un navegador tenia el stream de video abierto."""

import socket
import threading
import time

import httpx

from servicio_mrz.app import SHUTDOWN_TIMEOUT_S, create_app, make_server
from servicio_mrz.config import Settings


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_shutdown_does_not_wait_for_an_open_video_stream():
    port = _free_port()
    settings = Settings(camera_enabled=False, port=port, _env_file=None)
    server = make_server(create_app(settings), settings)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started

    # Un cliente (el navegador con /probar) mantiene el stream abierto durante todo el apagado.
    with httpx.stream("GET", f"http://127.0.0.1:{port}/camara/stream", timeout=None):
        time.sleep(0.5)
        requested = time.monotonic()
        server.should_exit = True  # lo mismo que hace Ctrl+C
        thread.join(timeout=SHUTDOWN_TIMEOUT_S + 5)
        elapsed = time.monotonic() - requested
        assert not thread.is_alive(), "el servidor sigue esperando al stream abierto"
    assert elapsed < SHUTDOWN_TIMEOUT_S + 3


def test_shutdown_is_clean_with_stream_and_websocket_open(caplog):
    """Regresion: con /probar abierto salian 'Cancel running tasks' y CancelledError como ERROR."""
    import logging

    import websockets.sync.client as ws_client

    caplog.set_level(logging.ERROR)
    port = _free_port()
    settings = Settings(camera_enabled=False, port=port, _env_file=None)
    server = make_server(create_app(settings), settings)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started

    with (
        httpx.stream("GET", f"http://127.0.0.1:{port}/camara/stream", timeout=None),
        ws_client.connect(f"ws://127.0.0.1:{port}/eventos") as sock,
    ):
        sock.recv(timeout=5)  # evento de estado inicial
        time.sleep(0.3)
        requested = time.monotonic()
        server.should_exit = True
        thread.join(timeout=SHUTDOWN_TIMEOUT_S + 5)
        elapsed = time.monotonic() - requested
        assert not thread.is_alive()
    assert elapsed < SHUTDOWN_TIMEOUT_S, f"cierre lento: {elapsed:.1f}s (cancelacion forzada)"
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert not errors, errors
