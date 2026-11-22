"""Registro por peticion de POST /mrz/leer: que mando la app y que se le devolvio.

Sirve para saber si un fallo es del servicio o de la app. NUNCA incluye valores del documento
(nombres, numeros, fechas): solo codigos, tipos, tamanos, tiempos y QUE campos van vacios.

Cada peticion lleva un identificador (`[0007]`) en todas sus lineas, incluidas las de lectura
(`leer: ...`). Se busca con:  Select-String -Path data\\servicio.log -Pattern "peticion:|respuesta:"
"""

from __future__ import annotations

import contextvars
import itertools
import json
import logging
from typing import Any

_rid: contextvars.ContextVar[str] = contextvars.ContextVar("rid", default="-")
_ids = itertools.count(1)

APP_TIMEOUT_S = 45  # la app (Dio) deja de esperar a los 45 s
NEAR_LIMIT_S = 40
FIELDS = ("firstNames", "lastNames", "documentNumber", "nationality", "birthDate")
LOG_FORMAT = "%(asctime)s %(levelname)s [%(rid)s] %(message)s"


class RidFilter(logging.Filter):
    """Pone el identificador de la peticion en curso en cada linea de log."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.rid = _rid.get()
        return True


def install() -> None:
    """Formato con hora e identificador para los logs del servicio (llamar al arrancar)."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    for handler in logging.getLogger().handlers:
        handler.addFilter(RidFilter())


def start() -> contextvars.Token[str]:
    return _rid.set(f"{next(_ids):04d}")


def finish(token: contextvars.Token[str]) -> None:
    _rid.reset(token)


def describe_request(
    *, photos: int, size: int | None, app_type: str | None, has_category: bool, agent: str
) -> str:
    return (
        f"peticion: fotos={photos} bytes={size if size is not None else '?'} "
        f"app_tipo={app_type or '-'} app_categoria={'presente' if has_category else 'ausente'} "
        f"agente={agent or '-'}"
    )


def _payload(body: Any) -> dict[str, Any]:
    if isinstance(body, dict):
        return body
    raw = getattr(body, "body", None)  # JSONResponse
    try:
        data = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        data = {}
    return data if isinstance(data, dict) else {}


def describe_response(
    status: int,
    body: Any,
    *,
    ms: int,
    app_type: str | None = None,
    app_disconnected: bool = False,
) -> str:
    """Una linea con lo que recibe la app. Solo tipos, codigos y campos vacios, sin valores."""
    data = _payload(body)
    parts = [f"respuesta: http={status}"]
    if status == 200:
        fields = data.get("fields") or {}
        empty = [name for name in FIELDS if not fields.get(name)]
        got = data.get("documentType")
        parts += [
            f"tipo={got}",
            f"modelo={data.get('documentModel')}",
            f"fuente={data.get('source')}",
            f"categoria={data.get('suggestedCategory')}",
            f"vacios={empty or '-'}",
            f"manual={data.get('requiresManualEntry', False)}",
            f"estado={data.get('status', '-')}",
            f"razones={data.get('reasons') or '-'}",
            # La app elige el tipo por la categoria de la entrada: si no coincide con lo detectado,
            # la validacion de la app puede rechazarlo aunque la lectura sea correcta.
            f"tipo_app_coincide={'-' if not app_type else ('si' if app_type == got else 'NO')}",
        ]
    else:
        parts += [
            f"codigo={data.get('codigo', '-')}",
            f"pista={data.get('pista', '-')}",
            f"origen={data.get('origen', '-')}",
            f"razones={data.get('reasons') or '-'}",
        ]
    parts.append(f"total_ms={ms}")
    if ms >= NEAR_LIMIT_S * 1000:
        parts.append(f"CERCA_DEL_CORTE_DE_{APP_TIMEOUT_S}s")
    # Si la app ya cerro la conexion, no vio esta respuesta: el fallo es de tiempo, no de lectura.
    parts.append(f"app_desconecto={'SI' if app_disconnected else 'no'}")
    return " ".join(parts)
