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
import threading
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
    Form,
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

# Si la 1a foto ya mostro una MRZ y gasto este tiempo sin leerla, no se intenta la 2a.
OLD_CI_MIN_S = 10.0  # trabajo minimo que se da al OCR del carnet antiguo
KYC_MIN_S = 5.0  # margen minimo que debe quedar para consultar el KYC
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
    reader = MrzReader(
        threads=settings.ocr_threads,
        budget_s=settings.ocr_budget_s,
        mem_arena=settings.ocr_mem_arena,
    )
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
        front: Annotated[UploadFile | None, File()] = None,
        back: Annotated[UploadFile | None, File()] = None,
        documentType: Annotated[str | None, Form()] = None,  # noqa: N803  (nombre del contrato)
        ticketTypeId: Annotated[str | None, Form()] = None,  # noqa: N803
    ):
        """Lee la MRZ de una foto y devuelve el JSON del contrato de la app.

        - `front` (o su alias `file`) + `back` opcional, en una sola peticion, como la app.
          Se lee la primera cara; si no tiene MRZ y llego la otra, se lee esa.
        - `documentType` y `ticketTypeId` se aceptan por compatibilidad y se IGNORAN: el tipo se
          deduce del propio documento (MRZ de 2 o 3 lineas, proporcion de la tarjeta).
        - Sin archivos: el servicio toma la foto de su propia camara y la lee.
        200 = JSON del contrato. 422 = no se leyo, con `pista`: `sin_mrz` (se ve un documento pero
        sin MRZ: dar la vuelta a la tarjeta) o `no_legible` (mejorar luz/enfoque/posicion).
        Las fotos no se guardan ni se registran sus datos.
        """
        raws: list[bytes] = []  # bytes originales: solo se usan si hay que consultar el KYC
        first = front if front is not None else file
        if first is not None:
            raws.append(await _upload(first))
            if back is not None:
                raws.append(await _upload(back))
            images = [_decode(raw) for raw in raws]
        else:
            latest = camera.latest()
            if latest is None:
                raise HTTPException(503, {"codigo": "CAMARA_SIN_IMAGEN", "mensaje": "Sin cuadros"})
            images = [latest[1]]
            raws = [encode_jpeg(latest[1], quality=90)]
        started = time.monotonic()
        card_seen = False
        mrz_seen = False  # alguna foto mostro una MRZ (aunque no se leyo): no es carnet antiguo
        mrz_ms: list[str] = []  # una entrada por foto leida (solo tiempos y forma, nunca datos)
        passport_shaped = False  # alguna tarjeta localizada mide menos de 1.47 de proporcion
        # La app SIEMPRE manda dos fotos ("frente" y "reverso") y NO dice que documento es: el tipo
        # se deduce de lo que se ve. Donde esta la MRZ depende del documento (cedula 2023: reverso;
        # pasaporte: pagina de datos, en cualquiera de las dos). El orden es solo una preferencia
        # y NUNCA se descarta una foto sin mirarla: la 2a se salta solo si la 1a ya mostro una MRZ
        # (misma pagina, otra captura) y gasto mucho tiempo.
        order = [images[1], images[0]] if len(images) == 2 else list(images)
        for img in order:
            # Nunca se descarta una foto por "haber visto una MRZ": esa senal es poco fiable (hay
            # carnets antiguos que la disparan) y la MRZ puede estar en cualquiera de las dos. Solo
            # se corta si ya no queda tiempo para los respaldos.
            if mrz_ms and time.monotonic() - started >= settings.request_deadline_s - 15:
                break
            t0 = time.monotonic()
            reading = await asyncio.to_thread(reader.read_detailed, img)
            # Para el log: tiempo, por donde se leyo (tarjeta localizada o OCR de todo el cuadro,
            # que es lo lento), proporcion de la tarjeta y si se vio texto con forma de MRZ.
            mrz_ms.append(
                f"{int((time.monotonic() - t0) * 1000)}ms "
                + (f"tarjeta:{reading.card_aspect:.2f}" if reading.card else "cuadro")
                + (" mrz" if reading.observation is not None else "")
            )
            if reading.checks_ok and reading.result is not None:
                ms = int((time.monotonic() - started) * 1000)
                # Solo tiempos y tamanos: nunca valores del documento.
                log.info(
                    "leer: fotos=%d %s mrz_ms=%s total_ms=%d resultado=mrz",
                    len(images),
                    [f"{i.shape[1]}x{i.shape[0]}" for i in images],
                    mrz_ms,
                    ms,
                )
                samaipateno = None
                if (
                    settings.birthplace_ocr
                    and contract.document_type(reading.result) == "CI_BO"
                    and time.monotonic() - started < settings.request_budget_s
                ):
                    try:
                        samaipateno = await asyncio.to_thread(reader.read_birthplace, img)
                    except Exception:  # noqa: BLE001 - un extra: nunca debe tumbar la lectura
                        log.warning("leer: no se pudo leer el lugar de nacimiento")
                    ms = int((time.monotonic() - started) * 1000)
                return contract.from_mrz(reading.result, elapsed_ms=ms, samaipateno=samaipateno)
            # tarjeta tipo cedula (ID-1) localizada sin MRZ valida: probablemente es el anverso.
            # Un pasaporte (~1.42) no tiene otra cara que mostrar: ahi el problema es la lectura.
            mrz_seen = mrz_seen or reading.observation is not None
            looks_id1 = reading.card is not None and reading.card_aspect >= ID1_MIN_ASPECT
            card_seen = card_seen or looks_id1
            passport_shaped = passport_shaped or (
                reading.card is not None and 0 < reading.card_aspect < ID1_MIN_ASPECT
            )
        # La proporcion discrimina mal (cedulas 1.46-1.61, pasaportes 1.26-1.47): una sola foto
        # "baja" no basta. Es pasaporte solo si NINGUNA foto mostro proporcion de cedula.
        passport_like = passport_shaped and not card_seen
        if settings.save_failed_photos:
            await asyncio.to_thread(_save_failed, Path(settings.data_dir) / "fallidas", raws)
        # Cedula antigua (sin MRZ): lectura local por plantilla (nombre y fecha del reverso, numero
        # del anverso). Solo si ninguna foto mostro una MRZ ni tiene forma de pasaporte.
        if settings.old_ci_ocr and not (mrz_seen or passport_like):
            t0 = time.monotonic()
            # Al menos 10 s de trabajo aunque la MRZ haya gastado mucho, pero nunca mas alla de
            # `old_ci_deadline_s` desde el inicio: la app abandona a los 45 s.
            old_deadline = min(
                max(t0 + OLD_CI_MIN_S, started + settings.old_ci_deadline_s),
                started + settings.request_deadline_s - 5,
            )
            old = await asyncio.to_thread(reader.read_old_ci, images, old_deadline)
            ms = int((time.monotonic() - started) * 1000)
            log.info(
                "leer: fotos=%d mrz_ms=%s carnet_antiguo_ms=%d total_ms=%d resultado=%s",
                len(images), mrz_ms, int((time.monotonic() - t0) * 1000), ms,
                "ocr_carnet_antiguo" if old else "carnet_antiguo_no_leido",
            )
            if old is not None:
                old["kycMs"] = old["duracionMs"] = ms
                return old
        # Sin MRZ legible: campos impresos por OCR (sin digitos de control: va marcado REVIEW)
        visual_ms: list[int] = []
        for img in images if settings.visual_ocr else []:
            if time.monotonic() - started >= settings.request_budget_s:
                log.warning("leer: presupuesto de %.0f s agotado, se omite el OCR impreso",
                            settings.request_budget_s)
                break  # un OCR impreso en un equipo lento puede tardar decenas de segundos
            t0 = time.monotonic()
            printed = await asyncio.to_thread(reader.read_visual, img)
            visual_ms.append(int((time.monotonic() - t0) * 1000))
            if printed is not None:
                ms = printed["kycMs"] = printed["duracionMs"] = int(
                    (time.monotonic() - started) * 1000
                )
                log.info(
                    "leer: fotos=%d %s mrz_ms=%s visual_ms=%s total_ms=%d resultado=ocr",
                    len(images),
                    [f"{i.shape[1]}x{i.shape[0]}" for i in images],
                    mrz_ms,
                    visual_ms,
                    ms,
                )
                return printed
        # Respaldo externo: cedulas antiguas (sin MRZ) y lecturas que el servicio local no logra.
        # Las fotos solo salen del totem hacia el KYC configurado en `.env`, y solo en este caso.
        # El KYC es SOLO para documentos que no tienen MRZ (la cedula antigua). No lee pasaportes
        # (comprobado en el totem) y un documento con MRZ que aqui no se pudo leer (pasaporte o
        # cedula 2023) necesita otra foto, no el KYC. "Tiene MRZ" = la lectura local vio texto
        # con estructura de MRZ (medido: ninguna foto sin MRZ lo mostro, todos los pasaportes si).
        time_left = settings.request_deadline_s - (time.monotonic() - started)
        if (
            settings.kyc_fallback
            and settings.kyc_base_url
            and not (passport_like or mrz_seen)
            and time_left >= KYC_MIN_S  # sin margen la llamada no puede acabar antes del corte
        ):
            # La app corta la espera a los 45 s: el servicio debe contestar antes (aunque sea con
            # KYC_TIEMPO_AGOTADO) para que la app muestre su aviso en vez de abandonar la peticion.
            remaining = settings.request_deadline_s - (time.monotonic() - started)
            kyc_settings = settings.model_copy(
                update={"kyc_timeout_s": min(settings.kyc_timeout_s, max(remaining, 5.0))}
            )
            try:
                # httpx aplica su timeout por operacion, no al total: wait_for pone el tope real.
                data = await asyncio.wait_for(
                    kyc_client.read_document(
                        kyc_settings, raws[0], raws[1] if len(raws) > 1 else None
                    ),
                    timeout=max(remaining, 5.0),
                )
            except (kyc_client.KycError, TimeoutError) as e:
                ms = int((time.monotonic() - started) * 1000)
                if isinstance(e, kyc_client.KycError):
                    code, message, kyc_elapsed = e.code, str(e), e.elapsed_ms
                else:
                    code, message, kyc_elapsed = (
                        "KYC_TIEMPO_AGOTADO", "El KYC no respondio a tiempo", None,
                    )
                log.info(
                    "leer: fotos=%d mrz_ms=%s total_ms=%d resultado=kyc_error codigo=%s",
                    len(images), mrz_ms, ms, code,
                )
                return JSONResponse(
                    {
                        "codigo": code,
                        "mensaje": message,
                        "reasons": ["mrz_not_found"],
                        "origen": "KYC",
                        "kycMs": kyc_elapsed or ms,
                        "duracionMs": kyc_elapsed or ms,
                    },
                    status_code=422 if code == "DOCUMENTO_NO_LEGIBLE" else 503,
                )
            ms = int((time.monotonic() - started) * 1000)
            kyc_ms = data.get("kycMs")
            data["duracionMs"] = data["kycMs"] = ms
            # Solo estado y codigos de razon (nunca valores del documento).
            log.info(
                "leer: fotos=%d mrz_ms=%s kyc_ms=%s total_ms=%d resultado=kyc estado=%s "
                "manual=%s razones=%s",
                len(images), mrz_ms, kyc_ms, ms, data.get("status"),
                data.get("requiresManualEntry"), data.get("reasons"),
            )
            return data
        ms = int((time.monotonic() - started) * 1000)
        log.info(
            "leer: fotos=%d %s mrz_ms=%s visual_ms=%s total_ms=%d resultado=no_legible",
            len(images),
            [f"{i.shape[1]}x{i.shape[0]}" for i in images],
            mrz_ms,
            visual_ms,
            ms,
        )
        hint = "sin_mrz" if card_seen else "no_legible"
        message = (
            "No se leyo la MRZ: de la vuelta a la tarjeta o mejore la luz y el enfoque"
            if hint == "sin_mrz"
            else "No se pudo leer la MRZ: mejore la luz, el enfoque o la posicion"
        )
        # `codigo` es el que la app ya sabe tratar (reintento); el detalle va en `reasons`/`pista`.
        return JSONResponse(
            {
                "codigo": "DOCUMENTO_NO_LEGIBLE",
                "mensaje": message,
                "reasons": ["mrz_not_found"],
                "pista": hint,
                "origen": "LOCAL",
                "kycMs": ms,
                "duracionMs": ms,
            },
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
    app.state.reader = reader
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


def _save_failed(folder: Path, raws: list[bytes]) -> None:
    """Guarda las fotos de una lectura fallida para medirlas despues (solo con SAVE_FAILED_PHOTOS).

    Son fotos de documentos reales: quedan solo en este equipo (data/ esta en .gitignore).
    """
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, raw in enumerate(raws):
        (folder / f"{stamp}_{i}.jpg").write_bytes(raw)
    log.info("leer: %d foto(s) fallida(s) guardadas en %s", len(raws), folder)


_PRIORITY_CLASS = {"below_normal": 0x4000, "normal": 0x20, "above_normal": 0x8000}


def _set_priority(level: str) -> None:
    """Fija la prioridad del proceso (solo Windows). Subir hasta 'above_normal' no pide admin.

    La tarea programada de Windows arranca el servicio con prioridad por DEBAJO de lo normal y la
    hereda: en el totem la app (pantalla y camara) le quita la CPU y la lectura pasa de ~3 s a
    ~15 s. Por eso se fija explicitamente en vez de dejarla como llegue.
    """
    try:
        import ctypes
        from ctypes import wintypes

        # Sin declarar los tipos, el pseudo-handle -1 de GetCurrentProcess se trunca a 32 bits y
        # la llamada falla en silencio.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), _PRIORITY_CLASS[level]):
            raise OSError(ctypes.get_last_error(), "SetPriorityClass")
        log.info("Prioridad del proceso: %s", level)
    except Exception:  # noqa: BLE001 - no es critico (otro sistema o sin permiso)
        log.debug("no se pudo fijar la prioridad del proceso")


def main() -> None:
    settings = Settings()
    logging.basicConfig(level=logging.INFO)
    _set_priority(settings.priority or ("below_normal" if settings.low_priority else "normal"))
    app = create_app(settings)
    # Cargar los modelos ahora y no con la primera foto del visitante (en un equipo lento tarda).
    threading.Thread(target=app.state.reader.warm_up, name="warm-up", daemon=True).start()
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
