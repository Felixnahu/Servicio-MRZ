"""Mide que variante de preprocesado lee mas fotos de samples/ (no imprime datos personales).

Uso:  uv run python tests/measure_variants.py [carpeta]
"""

import sys
import time
from pathlib import Path

import cv2

from servicio_mrz import cardfinder, celldecode, mrzread, preprocess
from servicio_mrz.ocr import MrzReader

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    photos = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
    reader = MrzReader()
    engine = reader._engine_ready()
    recognizer = celldecode.LineRecognizer(engine)
    cards = []
    for photo in photos:
        img = cv2.imread(str(photo))
        located = cardfinder.locate_card(img) if img is not None else None
        if located is not None:
            cards.append(cardfinder.warp_card(img, located).image)
    print(f"fotos: {len(photos)}  con tarjeta localizada: {len(cards)}\n")
    combos = [(name,) for name in preprocess.VARIANTS] + [tuple(preprocess.VARIANTS)]
    for combo in combos:
        ok, started = 0, time.monotonic()
        for card in cards:
            hit = False
            for img in (card, cv2.rotate(card, cv2.ROTATE_180)):
                read = mrzread.read_card(engine, recognizer, img, combo)
                if read.result and read.result.core_checks_ok:
                    hit = True
                    break
            ok += hit
        secs = (time.monotonic() - started) / max(len(cards), 1)
        print(f"{'+'.join(combo):40} {ok}/{len(cards)}   {secs:5.2f} s/foto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
