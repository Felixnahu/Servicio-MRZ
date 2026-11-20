"""Camara elegida y su calibracion, guardadas por NOMBRE (los indices de Windows cambian)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
PROFILE = Path("camera_profile.json")  # gitignored


def save(name: str, width: int, height: int, path: Path = PROFILE, **extra: Any) -> None:
    data = {"name": name, "width": width, "height": height, **extra}
    path.write_text(json.dumps(data), encoding="utf-8")


def load(path: Path = PROFILE) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data.get("name"), str) else None
    except (OSError, ValueError, AttributeError):
        return None


def clear_calibration(path: Path = PROFILE) -> bool:
    """Quita los ajustes calibrados conservando la camara elegida."""
    profile = load(path)
    if not profile or "settings" not in profile:
        return False
    save(profile["name"], profile["width"], profile["height"], path)
    return True


def resolve_index(names: list[str], profile: dict | None, default: int) -> int:
    """Indice de la camara guardada si sigue conectada; si no, el predeterminado."""
    if profile:
        for index, name in enumerate(names):
            if name == profile["name"]:
                return index
        log.warning("La camara guardada ya no esta conectada; se usa la predeterminada")
    return default
