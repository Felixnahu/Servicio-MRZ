"""Compara el OCR en CPU y en GPU (DirectML) con una imagen sintetica (sin datos personales).

Se ejecuta en un entorno APARTE (ver probar_aceleracion.ps1): no toca el entorno del servicio.
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


def synthetic_card() -> np.ndarray:
    img = np.full((830, 1226, 3), 235, np.uint8)
    for i in range(25):
        text = f"CAMPO {i:02d} ABCDEFGHIJ 0123456789 SAMAIPATA"
        cv2.putText(img, text, (40 + (i % 2) * 560, 50 + (i // 2) * 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def make_engine(threads: int, dml: bool):
    from rapidocr_onnxruntime import RapidOCR
    from rapidocr_onnxruntime.main import DEFAULT_CFG_PATH, read_yaml

    config = read_yaml(DEFAULT_CFG_PATH)
    for section in ("Det", "Cls", "Rec"):
        config[section]["intra_op_num_threads"] = threads
        config[section]["inter_op_num_threads"] = 1
        config[section]["use_dml"] = dml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    try:
        return RapidOCR(config_path=f.name)
    finally:
        Path(f.name).unlink(missing_ok=True)


def providers_in_use() -> list[str]:
    import onnxruntime as ort

    found = set()
    for obj in gc.get_objects():
        if isinstance(obj, ort.InferenceSession):
            found.add(obj.get_providers()[0])
    return sorted(found)


def run(label: str, threads: int, dml: bool, img: np.ndarray) -> None:
    engine = make_engine(threads, dml)
    t = time.monotonic()
    first, _ = engine(img)  # incluye compilacion/carga en GPU
    t_first = time.monotonic() - t
    times = []
    for _ in range(3):
        t = time.monotonic()
        res, _ = engine(img)
        times.append(time.monotonic() - t)
    print(
        f"{label:12} proveedor={providers_in_use()}  1a vez {t_first:.1f} s | "
        f"despues mediana {statistics.median(times):.1f} s (min {min(times):.1f}, "
        f"max {max(times):.1f}) | cajas {len(res or [])}",
        flush=True,
    )


def main() -> None:
    import onnxruntime as ort

    print(f"python {sys.version.split()[0]} | onnxruntime {ort.__version__} | "
          f"nucleos {os.cpu_count()}")
    print(f"proveedores disponibles: {ort.get_available_providers()}", flush=True)
    img = synthetic_card()
    run("CPU 2 hilos", 2, False, img)
    if "DmlExecutionProvider" in ort.get_available_providers():
        run("GPU DirectML", 2, True, img)
    else:
        print("DirectML no esta disponible en este entorno: no se prueba la GPU.")


if __name__ == "__main__":
    main()
