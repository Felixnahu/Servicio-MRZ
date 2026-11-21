"""Tiempo y campos leidos en carnets antiguos (frente + reverso), sin imprimir datos personales.

Uso:  uv run python tests/measure_old_ci.py "C:/ruta/a/la/carpeta"
La carpeta debe tener pares "front*.jpg" / "back*.jpg" con el mismo sufijo
(front (1).jpg y back (1).jpg).
"""

import sys
import time
from pathlib import Path

import cv2

from servicio_mrz.config import Settings
from servicio_mrz.ocr import MrzReader


def main() -> int:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "test_photos/CI")
    reader = MrzReader(threads=Settings(_env_file=None).ocr_threads)
    reader.warm_up()
    total, read = 0.0, 0
    pairs = sorted(folder.glob("front*.jpg"))
    for front in pairs:
        back = front.with_name(front.name.replace("front", "back", 1))
        images = [cv2.imread(str(front))] + ([cv2.imread(str(back))] if back.exists() else [])
        started = time.monotonic()
        res = reader.read_old_ci(images)
        elapsed = time.monotonic() - started
        total += elapsed
        if res is None:
            print(f"{front.stem:12} {elapsed:4.1f}s  NO LEIDO")
            continue
        read += 1
        f = res["fields"]
        words = len((f["firstNames"] + " " + f["lastNames"]).split())
        number = "si" if f["documentNumber"] else "NO"
        print(f"{front.stem:12} {elapsed:4.1f}s fecha=si nombres={words} palabras numero={number}")
    mean = total / max(len(pairs), 1)
    print(f"\n{read}/{len(pairs)} pares con fecha de nacimiento | media {mean:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
