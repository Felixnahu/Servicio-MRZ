"""Contrato JSON que espera la app del totem (el mismo que devuelve el backend SIBOSAM).

La app solo lee `documentType`, `categoryMatches`, `kycMs` y `fields.{firstNames, lastNames,
documentNumber, nationality, birthDate}`; el resto puede ir en null. Los campos extra (`source`,
`mrzChecks`) son informativos y la app los ignora.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

from servicio_mrz.mrz import MrzResult

_BOLIVIA = timezone(timedelta(hours=-4))
ADULT_AGE = 18


def today_in_bolivia() -> date:
    return datetime.now(UTC).astimezone(_BOLIVIA).date()


def age_at(born: date, today: date) -> int:
    years = today.year - born.year
    return years - ((today.month, today.day) < (born.month, born.day))


def document_type(result: MrzResult) -> str:
    """Codigos de la app: CI_BO, FOREIGN_ID, PASSPORT."""
    if result.document_code.startswith("P"):
        return "PASSPORT"
    return "CI_BO" if result.issuing_state == "BOL" else "FOREIGN_ID"


def document_model(result: MrzResult) -> str:
    """Variante del documento. La cedula con MRZ es la de 2023 (la app busca '2023' en el texto)."""
    kind = document_type(result)
    return {"PASSPORT": "passport", "CI_BO": "ci-2023"}.get(kind, "foreign-id")


def suggested_category(
    nationality: str, age: int | None, samaipateno: bool | None = None
) -> str | None:
    """Categoria de entrada que sugiere el documento (`samaipateno`, `nacional_*`, `extranjero_*`).

    'samaipateno' solo si el lugar de nacimiento leido es Samaipata; sin esa lectura no se adivina.
    """
    if samaipateno:
        return "samaipateno"
    if age is None or not nationality:
        return None
    origin = "nacional" if nationality.upper() == "BOL" else "extranjero"
    return f"{origin}_{'adulto' if age >= ADULT_AGE else 'menor'}"


def from_mrz(
    result: MrzResult,
    *,
    elapsed_ms: int,
    today: date | None = None,
    samaipateno: bool | None = None,
) -> dict[str, Any]:
    today = today or today_in_bolivia()
    born = result.birth_date
    age = age_at(born, today) if born else None
    return {
        "documentType": document_type(result),
        "documentModel": document_model(result),
        "fields": {
            "firstNames": result.given_names,
            "lastNames": result.surname,
            "documentNumber": result.document_number,
            "nationality": result.nationality,
            "birthDate": born.isoformat() if born else "",
        },
        "age": age,
        "isAdult": None if age is None else age >= ADULT_AGE,
        "isSamaipateno": samaipateno,
        "suggestedCategory": suggested_category(result.nationality, age, samaipateno),
        "categoryMatches": None,
        "kycMs": elapsed_ms,
        "duracionMs": elapsed_ms,
        "source": "mrz",
        "mrzChecks": result.checks,
        "mrzFusedFrames": result.fused_frames,
    }
