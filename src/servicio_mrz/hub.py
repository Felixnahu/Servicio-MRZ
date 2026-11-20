"""Reparto de eventos a los clientes WebSocket (la app y la pagina de pruebas)."""

from __future__ import annotations

import asyncio
from typing import Any

MAX_PENDING = 50


class Hub:
    def __init__(self) -> None:
        self._clients: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_PENDING)
        self._clients.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._clients.discard(queue)

    def publish(self, message: dict[str, Any]) -> None:
        """Seguro desde cualquier hilo (el reconocedor corre fuera del bucle de eventos)."""
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._fan_out, message)

    def _fan_out(self, message: dict[str, Any]) -> None:
        for queue in list(self._clients):
            if queue.full():  # cliente lento: descarta lo mas viejo, nunca bloquea al reconocedor
                queue.get_nowait()
            queue.put_nowait(message)
