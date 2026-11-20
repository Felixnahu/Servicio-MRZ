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
