"""Localiza una tarjeta/pasaporte en el cuadro y la endereza (vista cenital, horizontal).

Adaptado de `kyc-ci-backend` (`docproc/engines/locate.py`, mismo autor del proyecto). Un modelo de
mapas de calor (DocAligner, Apache 2.0, ver models/NOTICE.md) propone las 4 esquinas; cada lado
se afina sobre el borde real (gradiente) y el cuadrilatero se verifica con evidencia que el modelo
no vio (bordes rectos, esquinas cerca de la propuesta, geometria de tarjeta).

Cambios respecto al original: acepta tarjetas giradas 90 grados (relacion de aspecto invertida) y
ofrece `warp_card`, que devuelve la imagen enderezada y la matriz para volver a las coordenadas
del cuadro original (para dibujar sobre el video).
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

_WORK_MAX_SIDE = 1600
_MODEL_PATH = Path(__file__).parent / "models" / "docaligner_lcnet100_h_e_bifpn_256_fp32.onnx"
_MODEL_SHA256 = "f4117b786e3a18470f3865c93f3c2bd69d9b998edd60f385574a5c665e79594e"
_MODEL_INPUT = 256
_HEATMAP_THRESHOLD = 0.3
_ASPECT_RANGE = (1.25, 2.0)  # ID-1 = 1.586; pasaporte (pagina de datos) = 1.42
_MAX_ANGLE_DEV = 20.0
_SIDE_INSET = 0.10
_SIDE_SAMPLES = 80
_SEARCH_RANGE = 0.03
_INLIER_TOLERANCE = 0.002
_MIN_INLIER_PX = 2.0
_MIN_SAMPLES = 8
_MIN_SIDE_SUPPORT = 0.5
_MAX_CORNER_DRIFT = 0.05
_APPROX_MARGIN = 0.02
_OPPOSITE_SIDE_RATIO = 1.3  # lados opuestos: con perspectiva moderada nunca difieren tanto
_FRAME_MIN_AREA = 0.5  # propuesta de marco completo: la tarjeta debe ocupar casi todo el cuadro
WARP_WIDTH = 1024


@dataclass(frozen=True)
class Located:
    corners: np.ndarray  # (4, 2) float32, pixeles del cuadro original, orden tl, tr, br, bl
    precision: Literal["precise", "approximate"]
    aspect: float  # lado largo / lado corto (>= 1)
    angle_dev: float


@dataclass(frozen=True)
class Warped:
    image: np.ndarray  # tarjeta enderezada, horizontal, WARP_WIDTH de ancho
    to_frame: np.ndarray  # homografia 3x3: coordenadas de `image` -> cuadro original
    located: Located


def order_corners(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32
    )


# --- propuesta del modelo ----------------------------------------------------------------------

_net: cv2.dnn.Net | None = None
_net_lock = threading.Lock()


def _load_net() -> cv2.dnn.Net:
    data = _MODEL_PATH.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != _MODEL_SHA256:
        raise RuntimeError(f"{_MODEL_PATH.name}: sha256 {digest}, esperado {_MODEL_SHA256}")
    net = cv2.dnn.readNetFromONNX(np.frombuffer(data, np.uint8))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    return net


def _heatmaps(bgr: np.ndarray) -> np.ndarray:
    global _net
    x = cv2.resize(bgr, (_MODEL_INPUT, _MODEL_INPUT), interpolation=cv2.INTER_LINEAR)
    blob = np.transpose(x, (2, 0, 1)).astype(np.float32)[None] / 255.0
    with _net_lock:  # cv2.dnn.Net conserva sus buffers: un forward a la vez
        if _net is None:
            _net = _load_net()
        _net.setInput(blob)
        return _net.forward()[0].copy()


def _propose(bgr: np.ndarray) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    corners = []
    for heat in _heatmaps(bgr)[:4]:
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(
            (heat >= _HEATMAP_THRESHOLD).astype(np.uint8)
        )
        if n <= 1:
            return None
        corners.append(centroids[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))])
    return order_corners(np.array(corners, dtype=np.float32))


# --- afinado de bordes -------------------------------------------------------------------------


@dataclass(frozen=True)
class _Side:
    point: np.ndarray
    direction: np.ndarray
    support: float


def _fit_line(pts: np.ndarray, dist_type: int) -> tuple[np.ndarray, np.ndarray]:
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), dist_type, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)


def _distance_to_line(pts: np.ndarray, point: np.ndarray, direction: np.ndarray) -> np.ndarray:
    rel = pts - point
    return np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])


def _refine_side(
    gx: np.ndarray, gy: np.ndarray, a: np.ndarray, b: np.ndarray, centre: np.ndarray
) -> _Side | None:
    h, w = gx.shape
    length = float(np.linalg.norm(b - a))
    if length < 1.0:
        return None
    u = (b - a) / length
    normal = np.array([-u[1], u[0]])
    if np.dot(normal, (a + b) / 2 - centre) < 0:
        normal = -normal
    ts = np.linspace(_SIDE_INSET, 1 - _SIDE_INSET, _SIDE_SAMPLES)
    reach = max(2, round(_SEARCH_RANGE * length))
    offsets = np.arange(-reach, reach + 1, dtype=np.float64)
    pts = a + ts[:, None, None] * (b - a) + offsets[None, :, None] * normal
    xs = np.rint(pts[..., 0]).astype(int)
    ys = np.rint(pts[..., 1]).astype(int)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    response = np.full(xs.shape, -1.0, dtype=np.float32)
    response[inside] = np.abs(
        gx[ys[inside], xs[inside]] * normal[0] + gy[ys[inside], xs[inside]] * normal[1]
    )
    k = response.argmax(axis=1)
    rows = np.arange(len(ts))
    valid = inside[rows, k] & (k > 0) & (k < len(offsets) - 1) & (response[rows, k] > 0)
    peaks = pts[rows[valid], k[valid]]
    if len(peaks) < _MIN_SAMPLES:
        return None
    tolerance = max(_MIN_INLIER_PX, _INLIER_TOLERANCE * length)
    point, direction = _fit_line(peaks, cv2.DIST_HUBER)
    inliers = peaks[_distance_to_line(peaks, point, direction) <= tolerance]
    if len(inliers) < _MIN_SAMPLES:
        return None
    point, direction = _fit_line(inliers, cv2.DIST_L2)
    on_line = _distance_to_line(peaks, point, direction) <= tolerance
    return _Side(point, direction, float(on_line.sum() / _SIDE_SAMPLES))


def _pushed_out(a: np.ndarray, b: np.ndarray, centre: np.ndarray) -> _Side:
    length = float(np.linalg.norm(b - a)) or 1.0
    u = (b - a) / length
    normal = np.array([-u[1], u[0]])
    if np.dot(normal, (a + b) / 2 - centre) < 0:
        normal = -normal
    return _Side(a + normal * _APPROX_MARGIN * length, u, 0.0)


# --- verificacion ------------------------------------------------------------------------------


def _intersect(p1, d1, p2, d2) -> np.ndarray | None:
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-6:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


def _geometry(quad: np.ndarray) -> tuple[float, float]:
    """(relacion lado largo / lado corto, mayor desviacion de 90 grados en una esquina)."""
    sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    horizontal, vertical = sides[0] + sides[2], max(sides[1] + sides[3], 1e-6)
    aspect = max(horizontal, vertical) / max(min(horizontal, vertical), 1e-6)
    devs = []
    for i in range(4):
        v1 = quad[i - 1] - quad[i]
        v2 = quad[(i + 1) % 4] - quad[i]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        devs.append(abs(np.degrees(np.arccos(np.clip(cos, -1, 1))) - 90))
    return float(aspect), float(max(devs))


def _verify(
    proposal: np.ndarray, sides: list[_Side], image_area: float, min_area_ratio: float
) -> tuple[np.ndarray, float, float] | None:
    corners = []
    for i in range(4):
        p = _intersect(
            sides[i - 1].point, sides[i - 1].direction, sides[i].point, sides[i].direction
        )
        if p is None:
            return None
        corners.append(p)
    quad = np.array(corners, dtype=np.float32)
    top, bottom = proposal[1] - proposal[0], proposal[2] - proposal[3]
    width = (np.linalg.norm(top) + np.linalg.norm(bottom)) / 2
    if np.linalg.norm(quad - proposal, axis=1).max() > _MAX_CORNER_DRIFT * max(width, 1.0):
        return None
    quad = order_corners(quad)
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return None
    if cv2.contourArea(quad) < image_area * min_area_ratio:
        return None
    sides = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    for a, b in ((sides[0], sides[2]), (sides[1], sides[3])):
        if max(a, b) > _OPPOSITE_SIDE_RATIO * max(min(a, b), 1e-6):
            return None
    aspect, dev = _geometry(quad)
    if not (_ASPECT_RANGE[0] <= aspect <= _ASPECT_RANGE[1] and dev <= _MAX_ANGLE_DEV):
        return None
    return quad, aspect, dev


# --- API ---------------------------------------------------------------------------------------


def locate_card(image: np.ndarray, min_area_ratio: float = 0.04) -> Located | None:
    h, w = image.shape[:2]
    scale = min(1.0, _WORK_MAX_SIDE / max(h, w))
    work = image
    if scale < 1.0:
        work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.GaussianBlur(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    wh, ww = work.shape[:2]
    area = float(wh * ww)

    proposal = _propose(work)
    found = (
        _verify_proposal(proposal, gx, gy, area, min_area_ratio) if proposal is not None else None
    )
    if found is None:
        # el modelo falla a veces con tarjetas que llenan el cuadro (fotos ya recortadas): se prueba
        # el marco completo, aceptado solo si los 4 lados se confirman con bordes reales
        frame = np.array([[0, 0], [ww - 1, 0], [ww - 1, wh - 1], [0, wh - 1]], np.float32)
        found = _verify_proposal(frame, gx, gy, area, _FRAME_MIN_AREA, need_all_sides=True)
    if found is None:
        return None
    quad, aspect, dev, confirmed = found
    return Located(quad / scale, "precise" if confirmed == 4 else "approximate", aspect, dev)


def _verify_proposal(
    proposal: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    area: float,
    min_area_ratio: float,
    need_all_sides: bool = False,
) -> tuple[np.ndarray, float, float, int] | None:
    centre = proposal.mean(axis=0)
    sides, confirmed = [], 0
    for i in range(4):
        a, b = proposal[i], proposal[(i + 1) % 4]
        side = _refine_side(gx, gy, a, b, centre)
        if side is not None and side.support >= _MIN_SIDE_SUPPORT:
            sides.append(side)
            confirmed += 1
        else:
            sides.append(_pushed_out(a, b, centre))
    if need_all_sides and confirmed < 4:
        return None
    verified = _verify(proposal, sides, area, min_area_ratio)
    if verified is None:
        return None
    return (*verified, confirmed)


def warp_card(image: np.ndarray, located: Located, width: int = WARP_WIDTH) -> Warped:
    """Tarjeta enderezada y horizontal (la orientacion 0/180 se resuelve al leer)."""
    tl, tr, br, bl = located.corners
    horizontal = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    vertical = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    corners = located.corners
    if (
        vertical > horizontal
    ):  # tarjeta girada 90 grados: reordena para que el lado largo sea el ancho
        corners = np.array([tr, br, bl, tl], dtype=np.float32)
    height = round(width / located.aspect)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC)
    return Warped(warped, np.linalg.inv(matrix), located)


def whole_image_as_card(image: np.ndarray) -> Warped | None:
    """Imagen que YA es la tarjeta (recorte ajustado/enderezado): se usa entera, a WARP_WIDTH.

    Solo si su proporcion es la de una cedula (1.585) o una pagina de pasaporte (1.42), con margen.
    """
    h, w = image.shape[:2]
    if not _ASPECT_RANGE[0] <= w / h <= 1.75:
        return None
    scale = WARP_WIDTH / w
    resized = cv2.resize(image, (WARP_WIDTH, round(h * scale)), interpolation=cv2.INTER_AREA)
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    located = Located(corners, "approximate", max(w / h, h / w), 0.0)
    to_frame = np.array([[1 / scale, 0, 0], [0, 1 / scale, 0], [0, 0, 1]], np.float64)
    return Warped(resized, to_frame, located)


def project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Puntos (N, 2) de la imagen enderezada -> cuadro original."""
    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float32), homography
    ).reshape(-1, 2)
