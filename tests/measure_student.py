"""Falsos positivos y aciertos del detector de carnet universitario, sin imprimir datos personales.

Uso:  uv run python tests/measure_student.py "<carpeta de negativos>" ["<carpeta de positivos>"]
Las carpetas se recorren con sus subcarpetas (jpg/jpeg). Solo se imprimen conteos y tiempos.
"""

import sys
import time
from pathlib import Path

import cv2

from servicio_mrz import student, visual
from servicio_mrz.ocr import MrzReader


def photos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))


def flagged(reader: MrzReader, path: Path) -> bool:
    img = cv2.imread(str(path))
    if img is None:
        return False
    engine = reader._engine_ready()
    card = visual._prepare(img)
    for rot in (card, cv2.rotate(card, cv2.ROTATE_180)):
        boxes, _, _ = visual.read_boxes(engine, rot, use_cls=False)
        if student.detect(boxes):
            return True
    return False


def main() -> int:
    reader = MrzReader(threads=2)
    reader.warm_up()
    for label, arg in (("negativos", 1), ("positivos", 2)):
        if len(sys.argv) <= arg:
            continue
        files = photos(Path(sys.argv[arg]))
        started = time.monotonic()
        hits = sum(flagged(reader, p) for p in files)
        print(
            f"{label}: {hits}/{len(files)} marcadas como universitario | "
            f"{(time.monotonic() - started) / max(len(files), 1):.1f} s/foto"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
