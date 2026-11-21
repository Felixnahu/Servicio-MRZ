from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from servicio_mrz import contract
from servicio_mrz.app import create_app
from servicio_mrz.config import Settings
from servicio_mrz.kyc_client import to_contract
from servicio_mrz.mrz import parse_mrz

TODAY = date(2026, 11, 20)
PASSPORT = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C<3UTO6908061F9406236ZE184226B<<<<<14",
]
ID_CARD = [
    "I<UTOD231458907<<<<<<<<<<<<<<<",
    "7408122F1204159UTO<<<<<<<<<<<6",
    "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
]


def test_passport_maps_to_app_contract():
    out = contract.from_mrz(parse_mrz(PASSPORT, today=TODAY), elapsed_ms=120, today=TODAY)
    assert out["documentType"] == "PASSPORT"
    assert out["fields"] == {
        "firstNames": "ANNA MARIA",
        "lastNames": "ERIKSSON",
        "documentNumber": "L898902C",
        "nationality": "UTO",
        "birthDate": "1969-08-06",
    }
    assert (out["age"], out["isAdult"]) == (57, True)
    assert out["isSamaipateno"] is None and out["suggestedCategory"] is None
    assert out["categoryMatches"] is None and out["kycMs"] == 120


def test_foreign_id_card_type():
    out = contract.from_mrz(parse_mrz(ID_CARD, today=TODAY), elapsed_ms=1, today=TODAY)
    assert out["documentType"] == "FOREIGN_ID"


def test_age_boundaries_use_birthday():
    assert contract.age_at(date(2008, 11, 21), TODAY) == 17
    assert contract.age_at(date(2008, 11, 20), TODAY) == 18


def test_external_response_is_translated_to_same_contract():
    out = to_contract(
        {
            "document_type": "CI_NACIONAL",
            "document_model": "ci_2023",
            "document_number": "1234567",
            "first_name": "ANA MARIA",
            "last_name": "PEREZ LOPEZ",
            "nationality": "BOL",
            "birth_date": "2001-06-10",
            "age": 25,
            "is_adult": True,
            "is_samaipateno": False,
            "suggested_category": "nacional_adulto",
            "requires_manual_entry": False,
            "status": "verified",
            "optical_diagnostics": {"reasons": []},
        },
        elapsed_ms=3000,
    )
    assert out["documentType"] == "CI_BO"
    assert out["fields"]["firstNames"] == "ANA MARIA"
    assert out["suggestedCategory"] == "nacional_adulto" and out["kycMs"] == 3000
    assert out["source"] == "external"


def test_unknown_category_becomes_null():
    out = to_contract({"suggested_category": "otra"}, elapsed_ms=1)
    assert out["suggestedCategory"] is None and out["documentType"] is None


def _client() -> TestClient:
    return TestClient(create_app(Settings(camera_enabled=False, kyc_base_url="", _env_file=None)))


def test_health_without_camera():
    with _client() as c:
        body = c.get("/health").json()
    assert body["status"] == "ok" and body["camera"]["running"] is False
    assert body["kycConfigured"] is False


def test_frame_without_camera_is_503():
    with _client() as c:
        assert c.get("/camara/frame").status_code == 503


def test_test_page_is_served():
    with _client() as c:
        res = c.get("/probar")
    assert res.status_code == 200 and "Tomar foto y leer" in res.text


def test_capture_without_camera_is_503():
    with _client() as c:
        assert c.post("/captura/leer").status_code == 503


def test_quality_metrics_flag_glare_and_blur():
    import numpy as np

    from servicio_mrz.app import quality

    white = np.full((100, 100, 3), 255, np.uint8)
    q = quality(white)
    assert q["reflejo_pct"] == 100.0 and q["nitidez"] == 0.0


def test_mrz_endpoint_rejects_non_image():
    with _client() as c:
        res = c.post("/mrz/leer", files={"file": ("x.jpg", b"no-es-imagen", "image/jpeg")})
    assert res.status_code == 400


def test_kyc_endpoint_without_config_is_503_with_code():
    with _client() as c:
        res = c.post("/kyc/leer", files={"front": ("f.jpg", b"x", "image/jpeg")})
    assert res.status_code == 503
    assert res.json()["codigo"] == "KYC_NO_CONFIGURADO"


def test_camera_list_and_choose_validation(monkeypatch):
    monkeypatch.setattr("servicio_mrz.app.device_names", lambda: ["Laptop", "USB"])
    with _client() as c:
        listing = c.get("/camara/lista").json()
        assert [x["nombre"] for x in listing["camaras"]] == ["Laptop", "USB"]
        assert c.post("/camara/elegir?indice=7").status_code == 404
        assert c.post("/camara/elegir?indice=0&ancho=1280").status_code == 400
        assert c.post("/camara/elegir?indice=0&ancho=99999&alto=10").status_code == 400


def test_camera_profile_resolves_by_name_not_index(tmp_path):
    from servicio_mrz import camera_profile

    path = tmp_path / "p.json"
    camera_profile.save("USB", 1280, 720, path)
    profile = camera_profile.load(path)
    assert camera_profile.resolve_index(["Laptop", "USB"], profile, 0) == 1
    assert camera_profile.resolve_index(["USB", "Laptop"], profile, 0) == 0  # el orden cambio
    assert camera_profile.resolve_index(["Laptop"], profile, 0) == 0  # ya no esta conectada
    assert camera_profile.load(tmp_path / "no_existe.json") is None


def _jpeg(img) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


FONT = Path("C:/Windows/Fonts/consola.ttf")
TD3_LINES = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C<3UTO6908061F9406236ZE184226B<<<<<14",
]


def _td3_jpeg() -> bytes:
    """MRZ TD3 sintetica (Consolas, NO OCR-B) sobre fondo blanco."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1500, 200), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONT), 44)
    for i, line in enumerate(TD3_LINES):
        draw.text((30, 30 + 70 * i), line, font=font, fill="black")
    return _jpeg(np.array(img)[:, :, ::-1].copy())


def test_app_sends_photo_and_gets_the_contract_json():
    """El flujo pedido: la app manda la foto y este servicio la reconoce y devuelve el JSON."""
    import pytest

    if not FONT.exists():
        pytest.skip("requiere la fuente Consolas de Windows")
    with _client() as c:
        res = c.post("/mrz/leer", files={"file": ("f.jpg", _td3_jpeg(), "image/jpeg")})
    assert res.status_code == 200
    body = res.json()
    assert body["source"] == "mrz" and body["documentType"] == "PASSPORT"
    assert body["fields"]["documentNumber"] == "L898902C"
    assert "mrzChecks" in body


def test_photo_without_mrz_is_422_with_a_hint():
    import numpy as np

    blank = _jpeg(np.full((480, 640, 3), 200, np.uint8))
    with _client() as c:
        res = c.post("/mrz/leer", files={"file": ("f.jpg", blank, "image/jpeg")})
    assert res.status_code == 422
    body = res.json()
    assert body["codigo"] == "DOCUMENTO_NO_LEGIBLE"  # el que la app de SIBOSAM sabe tratar
    assert body["reasons"] == ["mrz_not_found"] and body["duracionMs"] == body["kycMs"]
    assert body["pista"] in {"sin_mrz", "no_legible"}


def test_app_style_request_front_and_back_with_extra_fields_in_one_call():
    """Como la app de SIBOSAM: `front` + `back` + documentType/ticketTypeId en una peticion."""
    import numpy as np
    import pytest

    if not FONT.exists():
        pytest.skip("requiere la fuente Consolas de Windows")
    blank = _jpeg(np.full((480, 640, 3), 200, np.uint8))
    with _client() as c:
        res = c.post(
            "/mrz/leer",
            data={"documentType": "PASSPORT", "ticketTypeId": "x"},
            files={
                "front": ("front.jpg", blank, "image/jpeg"),
                "back": ("back.jpg", _td3_jpeg(), "image/jpeg"),
            },
        )
    assert res.status_code == 200
    body = res.json()
    assert body["fields"]["documentNumber"] == "L898902C" and "duracionMs" in body


def test_back_face_is_used_when_the_front_has_no_mrz():
    import numpy as np
    import pytest

    if not FONT.exists():
        pytest.skip("requiere la fuente Consolas de Windows")
    blank = _jpeg(np.full((480, 640, 3), 200, np.uint8))
    with _client() as c:
        res = c.post(
            "/mrz/leer",
            files={
                "file": ("a.jpg", blank, "image/jpeg"),
                "back": ("b.jpg", _td3_jpeg(), "image/jpeg"),
            },
        )
    assert res.status_code == 200
    assert res.json()["fields"]["documentNumber"] == "L898902C"


def test_without_files_it_uses_the_totem_camera_and_is_503_if_there_is_none():
    with _client() as c:
        res = c.post("/mrz/leer")
    assert res.status_code == 503
    assert res.json()["detail"]["codigo"] == "CAMARA_SIN_IMAGEN"


def test_test_page_has_the_send_photo_panel_wired_to_the_app_endpoint():
    with _client() as c:
        html = c.get("/probar").text
    for element in ("fFront", "fBack", "sendPhotos", "snap", "photoBanner", "photoRaw"):
        assert f'id="{element}"' in html
    assert 'fetch("/mrz/leer"' in html
    # la pagina ya no reconoce en vivo: sin WebSocket ni recuadros; pausa el reconocimiento continuo
    assert "WebSocket" not in html and 'id="ov"' not in html
    assert "/reconocimiento?auto=false" in html


def test_recognition_mode_can_be_switched_to_photo_only():
    with _client() as c:
        assert c.get("/reconocimiento").json() == {"auto": True}
        assert c.post("/reconocimiento", params={"auto": "false"}).json() == {"auto": False}
        assert c.get("/reconocimiento").json() == {"auto": False}
    photo_only = TestClient(
        create_app(Settings(camera_enabled=False, auto_recognition=False, _env_file=None))
    )
    with photo_only as c:
        assert c.get("/reconocimiento").json() == {"auto": False}


def _totem_client(monkeypatch, kyc_result):
    """Como el totem: sin OCR local, con KYC configurado (simulado: no sale ninguna foto)."""
    import numpy as np

    from servicio_mrz import kyc_client
    from servicio_mrz.ocr import MrzReader, Reading

    calls = []

    async def fake_read(settings, front, back=None):
        calls.append((len(front), None if back is None else len(back)))
        if isinstance(kyc_result, Exception):
            raise kyc_result
        return dict(kyc_result)

    monkeypatch.setattr(kyc_client, "read_document", fake_read)
    # hay lineas de texto en la foto: un documento sin MRZ legible, no una mesa vacia
    with_text = Reading(None, rows=(object(), object()))
    monkeypatch.setattr(MrzReader, "read_detailed", lambda self, img, *a, **k: with_text)
    settings = Settings(
        camera_enabled=False,
        visual_ocr=False,
        kyc_base_url="https://kyc.invalid",
        _env_file=None,
    )
    blank = _jpeg(np.full((400, 600, 3), 200, np.uint8))
    return TestClient(create_app(settings)), blank, calls


def test_without_mrz_the_photos_go_to_the_kyc_and_its_contract_is_returned(monkeypatch):
    kyc = {"documentType": "CI_BO", "fields": {"birthDate": "1990-03-15"}, "source": "external"}
    client, blank, calls = _totem_client(monkeypatch, kyc)
    with client as c:
        res = c.post(
            "/mrz/leer",
            data={"documentType": "CI_BO"},
            files={"front": ("f.jpg", blank), "back": ("b.jpg", blank)},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["source"] == "external" and body["fields"]["birthDate"] == "1990-03-15"
    assert "duracionMs" in body and calls == [(len(blank), len(blank))]


def test_kyc_errors_keep_the_codes_the_app_understands(monkeypatch):
    from servicio_mrz import kyc_client

    cases = [
        ("DOCUMENTO_NO_LEGIBLE", 422),
        ("KYC_TIEMPO_AGOTADO", 503),
        ("KYC_SIN_CONEXION", 503),
    ]
    for code, status in cases:
        client, blank, _ = _totem_client(monkeypatch, kyc_client.KycError(code, "x"))
        with client as c:
            res = c.post(
                "/mrz/leer", data={"documentType": "CI_BO"}, files={"front": ("f.jpg", blank)}
            )
        body = res.json()
        assert (res.status_code, body["codigo"], body["origen"]) == (status, code, "KYC")


def test_valid_mrz_never_reaches_the_kyc(monkeypatch):
    import pytest

    if not FONT.exists():
        pytest.skip("requiere la fuente Consolas de Windows")
    client, _, calls = _totem_client(monkeypatch, {})
    monkeypatch.undo()  # lectura MRZ real; el KYC sigue sin configurarse
    with _client() as c:
        res = c.post("/mrz/leer", files={"front": ("f.jpg", _td3_jpeg(), "image/jpeg")})
    assert res.status_code == 200 and res.json()["source"] == "mrz" and calls == []


def test_a_slow_kyc_is_cut_at_the_total_deadline(monkeypatch):
    """Aunque el KYC no conteste, el servicio responde antes de que la app abandone."""
    import asyncio
    import time

    import numpy as np

    from servicio_mrz import kyc_client
    from servicio_mrz.ocr import MrzReader, Reading

    async def never(settings, front, back=None):
        await asyncio.sleep(30)

    monkeypatch.setattr(kyc_client, "read_document", never)
    with_text = Reading(None, rows=(object(), object()))
    monkeypatch.setattr(MrzReader, "read_detailed", lambda self, img, *a, **k: with_text)
    settings = Settings(
        camera_enabled=False,
        visual_ocr=False,
        kyc_base_url="https://kyc.invalid",
        request_deadline_s=1.0,  # el minimo de espera al KYC es 5 s
        _env_file=None,
    )
    blank = _jpeg(np.full((400, 600, 3), 200, np.uint8))
    started = time.monotonic()
    with TestClient(create_app(settings)) as c:
        res = c.post(
            "/mrz/leer", data={"documentType": "CI_BO"}, files={"front": ("f.jpg", blank)}
        )
    assert res.status_code == 503 and res.json()["codigo"] == "KYC_TIEMPO_AGOTADO"
    assert time.monotonic() - started < 10


def test_passports_are_never_sent_to_the_kyc(monkeypatch):
    client, blank, calls = _totem_client(monkeypatch, {"source": "external"})
    with client as c:
        res = c.post(
            "/mrz/leer",
            data={"documentType": "PASSPORT"},
            files={"front": ("f.jpg", blank), "back": ("b.jpg", blank)},
        )
    assert res.status_code == 422 and res.json()["codigo"] == "DOCUMENTO_NO_LEGIBLE"
    assert calls == []


def test_failed_photos_are_saved_only_when_asked(monkeypatch, tmp_path):
    import numpy as np

    from servicio_mrz.ocr import MrzReader, Reading

    monkeypatch.setattr(MrzReader, "read_detailed", lambda self, img, *a, **k: Reading(None))
    blank = _jpeg(np.full((400, 600, 3), 200, np.uint8))

    def post(save: bool):
        settings = Settings(
            camera_enabled=False, visual_ocr=False, save_failed_photos=save,
            data_dir=str(tmp_path), _env_file=None,
        )
        with TestClient(create_app(settings)) as c:
            return c.post("/mrz/leer", files={"front": ("f.jpg", blank), "back": ("b.jpg", blank)})

    assert post(False).status_code == 422 and not (tmp_path / "fallidas").exists()
    assert post(True).status_code == 422
    assert len(list((tmp_path / "fallidas").glob("*.jpg"))) == 2


def test_passport_skips_the_second_photo_when_the_first_was_slow(monkeypatch):
    import time

    import numpy as np

    from servicio_mrz import app as app_module
    from servicio_mrz.ocr import MrzReader, Reading

    seen = []

    def slow(self, img, *a, **k):
        seen.append(1)
        time.sleep(0.3)
        return Reading(None)

    monkeypatch.setattr(MrzReader, "read_detailed", slow)
    monkeypatch.setattr(app_module, "SECOND_PHOTO_MAX_S", 0.1)
    blank = _jpeg(np.full((400, 600, 3), 200, np.uint8))
    files = {"front": ("f.jpg", blank), "back": ("b.jpg", blank)}
    for doc_type, expected in (("PASSPORT", 1), ("FOREIGN_ID", 2)):
        seen.clear()
        settings = Settings(camera_enabled=False, visual_ocr=False, _env_file=None)
        with TestClient(create_app(settings)) as c:
            c.post("/mrz/leer", data={"documentType": doc_type}, files=files)
        assert len(seen) == expected


def test_a_document_without_type_or_with_another_type_still_reaches_the_kyc(monkeypatch):
    """La cedula antigua no debe quedar fuera por el tipo: solo se excluyen los pasaportes."""
    for data in ({}, {"documentType": "CI_BO"}, {"documentType": "FOREIGN_ID"}):
        client, blank, calls = _totem_client(monkeypatch, {"source": "external"})
        with client as c:
            res = c.post("/mrz/leer", data=data, files={"front": ("f.jpg", blank)})
        assert res.status_code == 200 and len(calls) == 1, data


def test_a_two_line_mrz_seen_locally_blocks_the_kyc_even_without_the_type(monkeypatch):
    """Pasaporte que la app no marco como tal: la lectura local ve una MRZ de 2 lineas."""
    from servicio_mrz.ocr import MrzReader, Reading

    client, blank, calls = _totem_client(monkeypatch, {"source": "external"})
    fake = type("R", (), {"format": "TD3", "core_checks_ok": False})()  # lectura incompleta
    monkeypatch.setattr(
        MrzReader, "read_detailed", lambda self, img, *a, **k: Reading(fake, rows=(1, 2))
    )
    with client as c:
        res = c.post("/mrz/leer", files={"front": ("f.jpg", blank)})
    assert res.status_code == 422 and calls == []


def test_ci_tries_only_the_back_locally_and_then_goes_to_the_kyc(monkeypatch):
    """CI_BO con dos fotos: la MRZ solo esta en el reverso; si no se lee, no se gasta otro intento
    local con el anverso (en el totem cada intento cuesta ~10 s) y se pasa al KYC."""
    from servicio_mrz.ocr import MrzReader, Reading

    seen = []
    with_text = Reading(None, rows=(1, 2))
    client, blank, calls = _totem_client(monkeypatch, {"source": "external"})

    def count(self, img, *a, **k):
        seen.append(1)
        return with_text

    monkeypatch.setattr(MrzReader, "read_detailed", count)
    files = {"front": ("f.jpg", blank), "back": ("b.jpg", blank)}
    with client as c:
        res = c.post("/mrz/leer", data={"documentType": "CI_BO"}, files=files)
    assert res.status_code == 200 and len(seen) == 1 and len(calls) == 1
    seen.clear()
    with client as c:  # tipo desconocido: se mantienen los dos intentos
        c.post("/mrz/leer", data={"documentType": "FOREIGN_ID"}, files=files)
    assert len(seen) == 2
