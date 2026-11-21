"""Mide la velocidad real de esta maquina con una imagen sintetica (sin fotos ni datos personales).

Uso (en el totem, con la app cerrada y luego abierta en la pantalla de la camara):
    uv run --no-sync python -m servicio_mrz.diagnostico

Sirve para separar "la maquina es lenta" de "algo le quita CPU al servicio mientras lee".
"""

from __future__ import annotations

import os
import statistics
import time

import cv2
import numpy as np


def _synthetic_card() -> np.ndarray:
    """Tarjeta con ~25 renglones de texto: carga parecida a una cedula real."""
    img = np.full((830, 1226, 3), 235, np.uint8)
    for i in range(25):
        text = f"CAMPO {i:02d} ABCDEFGHIJ 0123456789 SAMAIPATA"
        cv2.putText(img, text, (40 + (i % 2) * 560, 50 + (i // 2) * 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def _cpu_score() -> float:
    """Segundos para un calculo fijo en un solo hilo (menos es mejor)."""
    best = 1e9
    for _ in range(3):
        t = time.monotonic()
        total = 0
        for i in range(3_000_000):
            total += i * i % 7
        best = min(best, time.monotonic() - t)
    return best


def main() -> None:
    from servicio_mrz.ocr import make_engine

    print(f"nucleos logicos: {os.cpu_count()}")
    print(f"calculo fijo (1 hilo): {_cpu_score():.2f} s")
    img = _synthetic_card()
    for threads in (1, 2, 4):
        engine = make_engine(threads)
        engine(img)  # calentamiento
        times = []
        for _ in range(3):
            t = time.monotonic()
            result, _ = engine(img)
            times.append(time.monotonic() - t)
        print(
            f"OCR con {threads} hilo(s): mediana {statistics.median(times):.1f} s "
            f"(min {min(times):.1f}, max {max(times):.1f}; {len(result or [])} cajas)"
        )


if __name__ == "__main__":
    main()
