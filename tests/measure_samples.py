"""Mide el acierto del lector MRZ sobre las fotos de samples/ (nunca imprime datos personales).

Uso:  uv run python tests/measure_samples.py [carpeta]      (por defecto: samples)
"""

import sys
import time
from pathlib import Path

import cv2

from servicio_mrz.ocr import MrzReader

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    photos = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not photos:
        print(f"No hay fotos en {folder.resolve()}")
        return 1
    reader = MrzReader()
    ok = 0
    print(f"{'foto':44} {'formato':8} {'num':4} {'nac':4} {'venc':5} {'comp':5} {'seg':>5}")
    for photo in photos:
        img = cv2.imread(str(photo))
        if img is None:
            print(f"{photo.name[:44]:44} ilegible")
            continue
        started = time.monotonic()
        result = reader.read(img)
        secs = time.monotonic() - started
        if result is None:
            print(f"{photo.name[:44]:44} {'SIN MRZ':8} {'':4} {'':4} {'':5} {'':5} {secs:5.1f}")
            continue
        marks = {k: ("ok" if v else "FALLA") for k, v in result.checks.items()}
        ok += result.core_checks_ok
        print(
            f"{photo.name[:44]:44} {result.format:8} {marks['document_number']:4} "
            f"{marks['birth_date']:4} {marks['expiry_date']:5} {marks['composite']:5} {secs:5.1f}"
        )
    print(f"\nLeidas con controles correctos (numero, nacimiento, vencimiento): {ok}/{len(photos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
