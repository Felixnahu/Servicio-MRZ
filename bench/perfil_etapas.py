"""Perfil por ETAPAS de una lectura, con las fotos reales guardadas (data\\fallidas).

Uso (en el totem, servicio detenido y app cerrada para no competir por CPU):
    uv run --no-sync python bench\\perfil_etapas.py [carpeta] [pares]   # data\\fallidas y 4 pares

Para cada foto mide por separado: localizar la tarjeta, enderezarla, decidir el sentido y la
lectura de MRZ en cada orientacion; y por cada par, el lector de carnet antiguo/universitario.
Solo imprime tiempos y formas (nunca nombres, numeros ni fechas). Sirve para ver DONDE se gastan los
12-16 s que el log marca como `mrz_ms` en documentos sin MRZ.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

from servicio_mrz import cardfinder, celldecode, mrzread, orientation
from servicio_mrz.ocr import MrzReader


def timed(fn, *args, **kwargs):
    started = time.monotonic()
    out = fn(*args, **kwargs)
    return out, time.monotonic() - started


def profile_photo(reader: MrzReader, engine, recognizer, label: str, img) -> None:
    h, w = img.shape[:2]
    located, t_loc = timed(_locate, img)
    note = f"tarjeta {located.aspect:.2f}" if located else "sin tarjeta"
    if located is None:
        print(f"  {label} {w}x{h} | localizar {t_loc:.1f}s ({note}): sin tarjeta no hay MRZ")
        return
    warped, t_warp = timed(cardfinder.warp_card, img, located)
    flipped, t_vote = timed(orientation.upside_down, engine, warped.image)
    first = cv2.rotate(warped.image, cv2.ROTATE_180) if flipped else warped.image
    second = warped.image if flipped else cv2.rotate(warped.image, cv2.ROTATE_180)
    read1, t_read1 = timed(mrzread.read_card, engine, recognizer, first, prefer_lines=3)
    _, t_read2 = timed(mrzread.read_card, engine, recognizer, second, prefer_lines=3)
    found = bool(read1.result and read1.result.core_checks_ok)
    print(
        f"  {label} {w}x{h} | localizar {t_loc:.1f}s ({note}) | enderezar {t_warp:.1f}s | "
        f"sentido {t_vote:.1f}s (al reves={flipped}) | MRZ 1.a {t_read1:.1f}s (valida={found}) | "
        f"MRZ 2.a {t_read2:.1f}s",
        flush=True,
    )


def _locate(img):
    try:
        return cardfinder.locate_card(img)
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/fallidas")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    fronts = sorted(folder.glob("*_0.jpg"))
    if len(fronts) > limit:
        fronts = [fronts[i * len(fronts) // limit] for i in range(limit)]
    if not fronts:
        raise SystemExit(f"No hay fotos *_0.jpg en {folder}")
    reader = MrzReader(threads=2)
    reader.warm_up()
    engine = reader._engine_ready()
    recognizer = celldecode.LineRecognizer(engine)
    for front in fronts:
        back = front.with_name(front.name.replace("_0.jpg", "_1.jpg"))
        imgs = [cv2.imread(str(p)) for p in (front, back) if p.exists()]
        print(front.name.replace("_0.jpg", ""), flush=True)
        for index, img in enumerate(imgs):
            profile_photo(reader, engine, recognizer, f"foto{index}", img)
        _, t_old = timed(reader.read_old_ci, imgs)
        print(f"  carnet antiguo/universitario (par completo): {t_old:.1f}s", flush=True)


if __name__ == "__main__":
    main()
