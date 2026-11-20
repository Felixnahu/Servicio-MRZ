"""Tiempo y aciertos por grupo de fotos (sin imprimir datos personales).

Uso:  uv run python tests/measure_time.py [carpeta_test]
Grupos: pasaportes y cedulas de samples/, y reversos/anversos de la carpeta de prueba.
"""

import sys
import time
from pathlib import Path

import cv2

from servicio_mrz.config import Settings
from servicio_mrz.ocr import MrzReader

EXT = {".jpg", ".jpeg", ".png", ".webp"}


def groups(test_dir: Path) -> dict[str, list[Path]]:
    samples = [p for p in sorted(Path("samples").glob("*")) if p.suffix.lower() in EXT]
    return {
        "pasaportes": [p for p in samples if p.name.startswith("passaporte")],
        "cedulas(samples)": [p for p in samples if not p.name.startswith("passaporte")],
        "reversos": sorted(test_dir.glob("back*")),
        "anversos": sorted(test_dir.glob("front*")),
    }


def main() -> int:
    test_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "test_photos")
    reader = MrzReader(threads=Settings(_env_file=None).ocr_threads)
    first = next(iter(groups(test_dir).values()))[0]
    reader.read_detailed(cv2.imread(str(first)))  # calentar los modelos (no cuenta)
    everything: list[float] = []
    print(f"{'grupo':18} {'leidas':>8} {'t.leen':>8} {'t.no leen':>10} {'max':>6} {'>3s':>5}")
    for name, files in groups(test_dir).items():
        ok_t, no_t, hits = [], [], 0
        for p in files:
            img = cv2.imread(str(p))
            t = time.monotonic()
            det = reader.read_detailed(img)
            dt = time.monotonic() - t
            everything.append(dt)
            (ok_t if det.checks_ok else no_t).append(dt)
            hits += bool(det.checks_ok)
        mean = lambda v: f"{sum(v) / len(v):.2f}s" if v else "-"  # noqa: E731
        allt = ok_t + no_t
        print(
            f"{name:18} {hits:>3}/{len(files):<4} {mean(ok_t):>8} {mean(no_t):>10} "
            f"{max(allt):>5.1f}s {sum(t > 3 for t in allt):>5}"
        )
    everything.sort()
    p90 = everything[int(len(everything) * 0.9) - 1]
    print(
        f"\ntotal {len(everything)} fotos | media {sum(everything) / len(everything):.2f}s | "
        f"p90 {p90:.2f}s | max {everything[-1]:.2f}s | >3s: {sum(t > 3 for t in everything)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
