"""API local del servicio (solo loopback).

Endpoints por ahora:
- GET  /health            estado del servicio y de la camara
- GET  /camara/stream     video en vivo (MJPEG) para que la app lo muestre
- GET  /camara/frame      un cuadro JPEG
- POST /mrz/leer          imagen -> JSON del contrato (lectura MRZ local)   [pruebas con fotos]
- POST /kyc/leer          front (+back) -> JSON del contrato via servicio externo
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from servicio_mrz import camera_profile, contract, kyc_client
from servicio_mrz.calibration import Calibrator
from servicio_mrz.camera import Camera, device_names, encode_jpeg
from servicio_mrz.config import Settings
from servicio_mrz.diagpage import DIAG_PAGE
from servicio_mrz.engine import Recognizer
from servicio_mrz.hub import Hub
from servicio_mrz.ocr import MrzReader
from servicio_mrz.testpage import PAGE

log = logging.getLogger("servicio_mrz")
MAX_UPLOAD = 10 * 1024 * 1024
STREAM_FPS = 10
STREAM_WIDTH = 960
ID1_MIN_ASPECT = 1.47  # medido: cedulas 1.47-1.61, pasaportes 1.26-1.46
SHUTDOWN_TIMEOUT_S = 2  # espera maxima a conexiones abiertas (el stream no termina solo)


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, {"codigo": "IMAGEN_ILEGIBLE", "mensaje": "No es una imagen"})
    return img


def quality(img: np.ndarray) -> dict[str, float]:
    """Metricas para diagnosticar por que una foto no se lee (no contienen datos del documento)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "nitidez": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "brillo": round(float(gray.mean()), 1),
        "reflejo_pct": round(float((gray >= 250).mean() * 100), 2),
    }


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "", name.lower())[:40]


async def _upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, {"codigo": "IMAGEN_DEMASIADO_GRANDE", "mensaje": "Mas de 10 MB"})
    return data


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    camera = Camera(settings.camera_index, settings.camera_width, settings.camera_height)
    reader = MrzReader(threads=settings.ocr_threads)
    calibrator = Calibrator(camera, reader, Path(settings.data_dir) / "calibracion")
    hub = Hub()

    def stopping() -> bool:
        """True cuando el servidor empezo a apagarse: streams y WebSocket terminan solos."""
        server = getattr(app.state, "server", None)
        return bool(server and server.should_exit)

    mode = {"auto": settings.auto_recognition}  # reconocimiento continuo si/no (modo foto)
    recognizer = Recognizer(
        camera, reader, hub.publish, paused=lambda: calibrator.running or not mode["auto"]
    )

    def require_technician(x_admin_token: Annotated[str | None, Header()] = None) -> None:
        """Si hay ADMIN_TOKEN configurado, la calibracion lo exige (no es para el visitante)."""
        expected = settings.admin_token
        if expected and not secrets.compare_digest(x_admin_token or "", expected):
            raise HTTPException(401, {"codigo": "NO_AUTORIZADO", "mensaje": "Token tecnico"})

    technician = Depends(require_technician)

    def save_calibration(result: dict) -> None:
        names = device_names()
        if 0 <= camera.index < len(names):
            width, height = result["resolucion"]
            camera_profile.save(
                names[camera.index],
                width,
                height,
                settings=result["ajustes"],
                validated=result["validada_con_mrz"],
                calibrated_at=datetime.now().isoformat(timespec="seconds"),
            )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.camera_enabled:
            profile = camera_profile.load()
            names = device_names()
            if profile:
                camera.width = profile.get("width", camera.width)
                camera.height = profile.get("height", camera.height)
            camera.index = camera_profile.resolve_index(names, profile, settings.camera_index)
            if not camera.start():
                log.warning("Servicio iniciado SIN camara")
            elif (
                profile
                and profile.get("settings")
                and names[camera.index : camera.index + 1] == [profile["name"]]
            ):
                accepted = camera.apply(profile["settings"])
                log.info("Calibracion aplicada: %s", accepted)
            hub.bind(asyncio.get_running_loop())
            recognizer.start()
        yield
        recognizer.stop()
        camera.stop()

    app = FastAPI(title="Servicio MRZ", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get("/estado")
    async def recognition_state():
        """Fase del reconocimiento y, si ya hay, el JSON del documento leido."""
        return recognizer.snapshot()

    @app.websocket("/eventos")
    async def events(ws: WebSocket):
        """Eventos en vivo: estado, resultado (el JSON), sin_mrz y retirado."""
        await ws.accept()
        queue = hub.subscribe()
        try:
            await ws.send_json({"tipo": "estado", **recognizer.snapshot()})
            while not stopping():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                except TimeoutError:
                    continue
                await ws.send_json(event)
            await ws.close()  # apagado ordenado: cierre limpio en vez de cancelacion forzada
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.unsubscribe(queue)

    @app.get("/reconocimiento")
    async def recognition_mode():
        """`auto`: la camara reconoce sola. Si es false (modo foto) solo se lee al pedirlo."""
        return {"auto": mode["auto"]}

    @app.post("/reconocimiento")
    async def set_recognition_mode(auto: bool):
        mode["auto"] = auto
        return {"auto": auto}

    @app.get("/diagnostico", response_class=HTMLResponse)
    async def diagnostics_page():
        return DIAG_PAGE

    @app.get("/calibracion/capacidades", dependencies=[technician])
    async def calibration_capabilities():
        if calibrator.running:
            raise HTTPException(409, {"codigo": "CALIBRANDO", "mensaje": "Calibracion en curso"})
        if not camera.running:
            raise HTTPException(503, {"codigo": "SIN_CAMARA", "mensaje": "Camara no disponible"})
        return await asyncio.to_thread(calibrator.capabilities)

    @app.post("/calibracion/iniciar", dependencies=[technician])
    async def calibration_start():
        if not camera.running:
            raise HTTPException(503, {"codigo": "SIN_CAMARA", "mensaje": "Camara no disponible"})
        if not calibrator.start(on_done=save_calibration):
            raise HTTPException(409, {"codigo": "CALIBRANDO", "mensaje": "Ya hay una en curso"})
        return {"estado": "running"}

    @app.get("/calibracion/estado", dependencies=[technician])
    async def calibration_state():
        s = calibrator.state
        return {
            "estado": s.status,
            "fase": s.phase,
            "progreso": s.progress,
            "log": s.log,
            "resultado": s.result,
            "perfil": camera_profile.load(),
        }

    @app.post("/calibracion/cancelar", dependencies=[technician])
    async def calibration_cancel():
        calibrator.cancel()
        return {"estado": "cancelando"}

    @app.post("/calibracion/restablecer", dependencies=[technician])
    async def calibration_reset():
        """Borra los ajustes calibrados y reabre la camara (algunos drivers conservan valores)."""
        if calibrator.running:
            raise HTTPException(409, {"codigo": "CALIBRANDO", "mensaje": "Calibracion en curso"})
        removed = camera_profile.clear_calibration()
        await asyncio.to_thread(camera.switch, camera.index)
        return {"restablecida": removed}

    @app.get("/camara/lista")
    async def list_cameras():
        names = await asyncio.to_thread(device_names)
        return {
            "camaras": [
                {"indice": i, "nombre": n, "activa": camera.running and i == camera.index}
                for i, n in enumerate(names)
            ],
            "resolucion": camera.resolution(),
        }

    @app.post("/camara/elegir")
    async def choose_camera(indice: int, ancho: int | None = None, alto: int | None = None):
        """Cambia la camara activa (y la resolucion pedida) y recuerda la eleccion por nombre."""
        names = await asyncio.to_thread(device_names)
        if not 0 <= indice < len(names):
            raise HTTPException(404, {"codigo": "CAMARA_NO_EXISTE", "mensaje": "Indice invalido"})
        if (ancho is None) != (alto is None) or any(
            v and not 160 <= v <= 4096 for v in (ancho, alto)
        ):
            raise HTTPException(400, {"codigo": "RESOLUCION_INVALIDA", "mensaje": "Ancho/alto"})
        ok = await asyncio.to_thread(camera.switch, indice, ancho, alto)
        if not ok:
            raise HTTPException(
                409, {"codigo": "CAMARA_NO_ABRE", "mensaje": f"No se pudo abrir {names[indice]}"}
            )
        width, height = camera.resolution() or (camera.width, camera.height)
        previous = camera_profile.load()
        kept = {}
        if previous and previous["name"] == names[indice] and "settings" in previous:
            kept = {"settings": previous["settings"]}  # misma camara: conserva su calibracion
        camera_profile.save(names[indice], camera.width, camera.height, **kept)
        return {"indice": indice, "nombre": names[indice], "resolucion": [width, height]}

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "camera": {"running": camera.running, "resolution": camera.resolution()},
            "kycConfigured": bool(settings.kyc_base_url and settings.kyc_api_key),
        }

    @app.get("/camara/frame")
    async def frame():
        latest = camera.latest()
        if latest is None:
            raise HTTPException(503, {"codigo": "CAMARA_SIN_IMAGEN", "mensaje": "Sin cuadros"})
        return Response(encode_jpeg(latest[1], quality=95), media_type="image/jpeg")

    @app.get("/camara/stream")
    async def stream():
        async def frames():
            last = -1
            while not stopping():
                latest = camera.latest()
                if latest and latest[0] != last:
                    last = latest[0]
                    jpg = encode_jpeg(latest[1], quality=70, max_width=STREAM_WIDTH)
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpg)).encode()
                        + b"\r\n\r\n"
                        + jpg
                        + b"\r\n"
                    )
                await asyncio.sleep(1 / STREAM_FPS)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/mrz/leer")
    async def read_mrz(
        file: Annotated[UploadFile | None, File()] = None,
        back: Annotated[UploadFile | None, File()] = None,
    ):
        """Lee la MRZ de una foto y devuelve el JSON del contrato de la app.

        - `file` (+ `back` opcional): la app manda la foto del documento (y la otra cara). Se lee
          la primera cara; si no tiene MRZ y llego la otra, se lee esa.
        - Sin archivos: el servicio toma la foto de su propia camara y la lee.
        200 = JSON del contrato. 422 = no se leyo, con `pista`: `sin_mrz` (se ve un documento pero
        sin MRZ: dar la vuelta a la tarjeta) o `no_legible` (mejorar luz/enfoque/posicion).
        Las fotos no se guardan ni se registran sus datos.
        """
        images: list[np.ndarray] = []
        if file is not None:
            images.append(_decode(await _upload(file)))
            if back is not None:
                images.append(_decode(await _upload(back)))
        else:
            latest = camera.latest()
            if latest is None:
                raise HTTPException(503, {"codigo": "CAMARA_SIN_IMAGEN", "mensaje": "Sin cuadros"})
            images.append(latest[1])
        started = time.monotonic()
        card_seen = False
        for img in images:
            reading = await asyncio.to_thread(reader.read_detailed, img)
            if reading.checks_ok and reading.result is not None:
                ms = int((time.monotonic() - started) * 1000)
                return contract.from_mrz(reading.result, elapsed_ms=ms)
            # tarjeta tipo cedula (ID-1) localizada sin MRZ valida: probablemente es el anverso.
            # Un pasaporte (~1.42) no tiene otra cara que mostrar: ahi el problema es la lectura.
            looks_id1 = reading.card is not None and reading.card_aspect >= ID1_MIN_ASPECT
            card_seen = card_seen or looks_id1
        # Sin MRZ legible: campos impresos por OCR (sin digitos de control: va marcado REVIEW)
        for img in images:
            printed = await asyncio.to_thread(reader.read_visual, img)
            if printed is not None:
                printed["kycMs"] = int((time.monotonic() - started) * 1000)
                return printed
        ms = int((time.monotonic() - started) * 1000)
        hint = "sin_mrz" if card_seen else "no_legible"
        message = (
            "No se leyo la MRZ: de la vuelta a la tarjeta o mejore la luz y el enfoque"
            if hint == "sin_mrz"
            else "No se pudo leer la MRZ: mejore la luz, el enfoque o la posicion"
        )
        return JSONResponse(
            {"codigo": "MRZ_NO_ENCONTRADA", "mensaje": message, "pista": hint, "kycMs": ms},
            status_code=422,
        )

    @app.get("/probar", response_class=HTMLResponse)
    async def test_page():
        return PAGE

    @app.post("/captura/leer")
    async def capture_and_read(guardar: bool = False, nombre: str = ""):
        """Toma el cuadro actual de la camara, lo lee y (opcional) lo guarda para medir despues."""
        latest = camera.latest()
        if latest is None:
            raise HTTPException(503, {"codigo": "CAMARA_SIN_IMAGEN", "mensaje": "Sin cuadros"})
        img = latest[1]
        info: dict = {"calidad": quality(img), "foto": None}
        if guardar:
            folder = Path(settings.samples_dir)
            folder.mkdir(parents=True, exist_ok=True)
            prefix = _safe_name(nombre) or "captura"
            file = folder / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(str(file), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            info["foto"] = file.name
        started = time.monotonic()
        result = await asyncio.to_thread(reader.read, img)
        ms = int((time.monotonic() - started) * 1000)
        if result is not None and result.core_checks_ok:
            return {**contract.from_mrz(result, elapsed_ms=ms), **info}
        attempt = None
        if result is not None:
            attempt = {"formato": result.format, "controles": result.checks}
        # Endpoint de diagnostico: no leer un cuadro es un resultado normal, no un error HTTP.
        return {
            "leido": False,
            "codigo": "MRZ_NO_ENCONTRADA",
            "mensaje": "Sin MRZ valida en el cuadro",
            "kycMs": ms,
            "intento": attempt,
            **info,
        }

    @app.post("/kyc/leer")
    async def read_kyc(
        front: Annotated[UploadFile, File()],
        back: Annotated[UploadFile | None, File()] = None,
    ):
        back_bytes = await _upload(back) if back else None
        try:
            return await kyc_client.read_document(settings, await _upload(front), back_bytes)
        except kyc_client.KycError as e:
            status = 422 if e.code == "DOCUMENTO_NO_LEGIBLE" else 503
            return JSONResponse(
                {"codigo": e.code, "mensaje": str(e), "origen": "KYC", "duracionMs": e.elapsed_ms},
                status_code=status,
            )

    app.state.camera = camera
    return app


def make_server(app: FastAPI, settings: Settings):
    """Servidor uvicorn que SI se apaga con Ctrl+C aunque haya un stream de video abierto.

    El stream es una respuesta infinita: sin `timeout_graceful_shutdown`, uvicorn espera a que
    el navegador la cierre y el proceso no termina nunca.
    """
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=settings.host,
            port=settings.port,
            timeout_graceful_shutdown=SHUTDOWN_TIMEOUT_S,
        )
    )
    app.state.server = server
    return server


def _lower_priority() -> None:
    """El servicio comparte maquina con la app del totem: que no le quite CPU (solo Windows)."""
    try:
        import ctypes

        below_normal = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), below_normal
        )
    except Exception:  # noqa: BLE001 - no es critico (otro sistema o sin permiso)
        log.debug("no se pudo bajar la prioridad del proceso")


def main() -> None:
    settings = Settings()
    _lower_priority()
    logging.basicConfig(level=logging.INFO)
    app = create_app(settings)
    try:
        make_server(app, settings).run()
    except KeyboardInterrupt:
        # uvicorn re-lanza Ctrl+C al terminar el apagado ordenado: no es un error, sin traceback.
        pass
    finally:
        # Si el apagado fue brusco (segundo Ctrl+C) el lifespan puede no haber corrido.
        app.state.camera.stop()


if __name__ == "__main__":
    main()
