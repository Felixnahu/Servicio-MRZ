"""Mide en ESTA maquina cuanto tarda nuestro lector con fotos reales y distintos repartos de hilos.

Uso (en el totem, con el servicio detenido para no competir por CPU):
    uv run --no-sync python bench/medir_hilos.py [carpeta]      # por defecto data\\fallidas

Lee los pares *_0.jpg / *_1.jpg (frente y reverso, como los manda la app). Solo imprime tiempos y si
se leyo o no: nunca nombres, numeros ni fechas.

Configuraciones (motores x hilos por motor):
  1x2  lo que usa hoy el servicio (un motor, 2 hilos), una foto tras otra
  1x4  un motor con los 4 nucleos, una foto tras otra
  2x2  dos motores con 2 hilos, las dos fotos a la vez (reparto tipo escanerci)
  4x1  cuatro motores con 1 hilo, las dos fotos a la vez
"""

from __future__ import annotations

import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from servicio_mrz.ocr import MrzReader


def _read_one(reader: MrzReader, img) -> bool:
    detail = reader.read_detailed(img)
    return bool(detail.checks_ok and detail.result is not None)


def _sequential(readers, imgs) -> bool:
    reader = readers[0]
    if any(_read_one(reader, img) for img in reversed(imgs)):
        return True
    return len(imgs) == 2 and reader.read_old_ci(imgs) is not None


def _parallel(readers, imgs) -> bool:
    with ThreadPoolExecutor(len(readers)) as pool:
        futures = [
            pool.submit(_read_one, readers[i % len(readers)], img) for i, img in enumerate(imgs)
        ]
        if any(f.result() for f in futures):
            return True
    return len(imgs) == 2 and readers[0].read_old_ci(imgs) is not None


CONFIGS = [
    ("1x2", 1, 2, _sequential),
    ("1x4", 1, 4, _sequential),
    ("2x2", 2, 2, _parallel),
    ("4x1", 4, 1, _parallel),
]


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/fallidas")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    fronts = sorted(folder.glob("*_0.jpg"))
    if len(fronts) > limit:
        fronts = [fronts[i * len(fronts) // limit] for i in range(limit)]
    pairs = []
    for front in fronts:
        back = front.with_name(front.name.replace("_0.jpg", "_1.jpg"))
        imgs = [cv2.imread(str(p)) for p in (front, back) if p.exists()]
        pairs.append([i for i in imgs if i is not None])
    if not pairs:
        raise SystemExit(f"No hay fotos *_0.jpg en {folder}")
    print(f"{len(pairs)} intentos en {folder}\n", flush=True)
    for name, engines, threads, run in CONFIGS:
        readers = [MrzReader(threads=threads) for _ in range(engines)]
        for r in readers:
            r.warm_up()
        times, ok = [], 0
        for imgs in pairs:
            t = time.monotonic()
            ok += run(readers, imgs)
            times.append(time.monotonic() - t)
        print(
            f"{name}: leidos {ok}/{len(pairs)} | mediana {statistics.median(times):.1f} s | "
            f"media {statistics.mean(times):.1f} s | max {max(times):.1f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
