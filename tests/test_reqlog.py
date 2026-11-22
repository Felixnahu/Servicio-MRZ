"""Registro por peticion: que llego de la app y que se devolvio. Nunca valores del documento."""

import logging

import numpy as np
from fastapi.testclient import TestClient

from servicio_mrz import reqlog
from servicio_mrz.app import create_app
from servicio_mrz.config import Settings
from servicio_mrz.ocr import MrzReader, Reading

OK = {
    "documentType": "CI_BO",
    "documentModel": "ci-2023",
    "fields": {
        "firstNames": "SECRETO",
        "lastNames": "PRIVADO",
        "documentNumber": "",
        "nationality": "BOL",
        "birthDate": "1990-01-01",
    },
    "age": 36,
    "suggestedCategory": "nacional_adulto",
    "source": "mrz",
}


def test_success_line_lists_empty_fields_but_never_values():
    line = reqlog.describe_response(200, OK, ms=1200, app_type="CI_BO")
    assert "tipo=CI_BO" in line and "modelo=ci-2023" in line and "categoria=nacional_adulto" in line
    assert "vacios=['documentNumber']" in line and "tipo_app_coincide=si" in line
    for secret in ("SECRETO", "PRIVADO", "1990", "36"):
        assert secret not in line.replace("total_ms=1200", "")


def test_it_flags_a_type_the_app_did_not_expect_and_a_late_answer():
    line = reqlog.describe_response(200, OK, ms=43000, app_type="PASSPORT", app_disconnected=True)
    assert "tipo_app_coincide=NO" in line
    assert "CERCA_DEL_CORTE_DE_45s" in line and "app_desconecto=SI" in line


def test_error_line_carries_the_code_the_app_sees():
    body = {"codigo": "DOCUMENTO_NO_LEGIBLE", "pista": "sin_mrz", "origen": "LOCAL"}
    line = reqlog.describe_response(422, body, ms=900)
    assert "http=422" in line and "codigo=DOCUMENTO_NO_LEGIBLE" in line
    assert "pista=sin_mrz" in line and "origen=LOCAL" in line and "app_desconecto=no" in line


def test_the_endpoint_logs_request_and_response_with_the_same_id(monkeypatch, caplog):
    import cv2

    monkeypatch.setattr(MrzReader, "read_detailed", lambda self, img, *a, **k: Reading(None))
    monkeypatch.setattr(MrzReader, "read_old_ci", lambda self, imgs, *a, **k: None)
    ok, buf = cv2.imencode(".jpg", np.full((400, 600, 3), 200, np.uint8))
    settings = Settings(camera_enabled=False, visual_ocr=False, _env_file=None)
    caplog.handler.addFilter(reqlog.RidFilter())  # lo instala main() al arrancar
    with caplog.at_level(logging.INFO, logger="servicio_mrz"), TestClient(
        create_app(settings)
    ) as c:
        c.post(
            "/mrz/leer",
            data={"documentType": "PASSPORT", "ticketTypeId": "abc"},
            files={"front": ("f.jpg", buf.tobytes()), "back": ("b.jpg", buf.tobytes())},
        )
    messages = [r.getMessage() for r in caplog.records]
    request = next(m for m in messages if m.startswith("peticion:"))
    response = next(m for m in messages if m.startswith("respuesta:"))
    assert "fotos=2" in request and "app_tipo=PASSPORT" in request
    assert "app_categoria=presente" in request and "abc" not in request  # el id no se registra
    assert "http=422" in response and "codigo=DOCUMENTO_NO_LEGIBLE" in response
    ids = {r.rid for r in caplog.records if r.getMessage().startswith(("peticion", "respuesta"))}
    assert len(ids) == 1 and ids != {"-"}
