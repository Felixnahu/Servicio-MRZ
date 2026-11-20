"""Cliente del servicio externo (docproc): `POST /v1/integrations/samaipata/validate`.

Convierte su respuesta al mismo contrato que la lectura MRZ. Nunca registra valores de campos ni
imagenes: solo codigos HTTP y tiempos.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from servicio_mrz.config import Settings

log = logging.getLogger(__name__)

_TYPES = {"CI_NACIONAL": "CI_BO", "CI_EXTRANJERO": "FOREIGN_ID", "PASAPORTE": "PASSPORT"}
_CATEGORIES = {
    "samaipateno",
    "nacional_menor",
    "nacional_adulto",
    "extranjero_menor",
    "extranjero_adulto",
}


class KycError(Exception):
    """Falla al consultar el KYC. `code` usa los mismos codigos que el backend SIBOSAM."""

    def __init__(self, code: str, message: str, *, elapsed_ms: int | None = None):
        super().__init__(message)
        self.code = code
        self.elapsed_ms = elapsed_ms


def to_contract(data: dict[str, Any], *, elapsed_ms: int) -> dict[str, Any]:
    category = data.get("suggested_category")
    return {
        "documentType": _TYPES.get(data.get("document_type") or ""),
        "documentModel": data.get("document_model"),
        "fields": {
            "firstNames": data.get("first_name") or "",
            "lastNames": data.get("last_name") or "",
            "documentNumber": data.get("document_number") or "",
            "nationality": data.get("nationality") or "",
            "birthDate": data.get("birth_date") or "",
        },
        "age": data.get("age"),
        "isAdult": data.get("is_adult"),
        "isSamaipateno": data.get("is_samaipateno"),
        "suggestedCategory": category if category in _CATEGORIES else None,
        "categoryMatches": None,
        "kycMs": elapsed_ms,
        "source": "external",
        "requiresManualEntry": bool(data.get("requires_manual_entry", False)),
        "status": data.get("status"),
        "reasons": (data.get("optical_diagnostics") or {}).get("reasons") or [],
    }


async def read_document(
    settings: Settings, front: bytes, back: bytes | None = None
) -> dict[str, Any]:
    if not settings.kyc_base_url:
        raise KycError("KYC_NO_CONFIGURADO", "Falta KYC_BASE_URL")
    files = {"front": ("front.jpg", front, "image/jpeg")}
    if back:
        files["back"] = ("back.jpg", back, "image/jpeg")
    headers = {"Authorization": f"Bearer {settings.kyc_api_key}"} if settings.kyc_api_key else {}
    url = settings.kyc_base_url.rstrip("/") + "/v1/integrations/samaipata/validate"
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=settings.kyc_timeout_s) as client:
            res = await client.post(url, files=files, headers=headers)
    except httpx.TimeoutException as e:
        ms = int((time.monotonic() - started) * 1000)
        raise KycError("KYC_TIEMPO_AGOTADO", "El KYC no respondio a tiempo", elapsed_ms=ms) from e
    except httpx.HTTPError as e:
        raise KycError("KYC_SIN_CONEXION", "No se pudo conectar con el KYC") from e
    ms = int((time.monotonic() - started) * 1000)
    log.info("KYC samaipata/validate: HTTP %s en %s ms", res.status_code, ms)
    if res.status_code == 400:
        raise KycError("DOCUMENTO_NO_LEGIBLE", "La imagen no se pudo leer", elapsed_ms=ms)
    if res.status_code == 413:
        raise KycError("IMAGEN_DEMASIADO_GRANDE", "La imagen supera el limite", elapsed_ms=ms)
    if res.status_code in (401, 403):
        raise KycError("KYC_CREDENCIALES_INVALIDAS", "El KYC rechazo la API key", elapsed_ms=ms)
    if res.status_code == 429:
        raise KycError("KYC_LIMITE_EXCEDIDO", "Demasiadas peticiones al KYC", elapsed_ms=ms)
    if not res.is_success:
        raise KycError("KYC_ERROR", f"El KYC respondio HTTP {res.status_code}", elapsed_ms=ms)
    return to_contract(res.json(), elapsed_ms=ms)
