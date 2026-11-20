"""Calibracion con camara y lector simulados: se conoce la respuesta correcta de la 'escena'."""

import numpy as np

from servicio_mrz.calibration import Calibrator

IDEAL_EXPOSURE = -3.0
IDEAL_FOCUS = 96.0


def _img(bright: float, amp: float) -> np.ndarray:
    arr = np.full((100, 100), bright - amp / 2, np.float64)  # el tablero suma amp/2 al promedio
    arr[::2, ::2] += amp
    arr[1::2, 1::2] += amp
    return np.repeat(np.clip(arr, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


class FakeCamera:
    """Escena: brillo sube 20 por paso de exposicion; nitidez es maxima con foco 96."""

    def __init__(self, supported: set[str], exposure_effective: bool = True):
        self.index, self.width, self.height = 0, 1280, 720
        self.values = {"exposure": -6.0, "focus": 0.0, "autofocus": 1.0, "brightness": 0.0}
        self.supported = supported
        self.exposure_effective = exposure_effective
        self.switched: list[tuple[int, int]] = []
        self.fake_fps = 30.0

    def get(self, name):
        return self.values.get(name, -1.0) if name in self.supported else -1.0

    def apply(self, settings):
        accepted = {}
        for name, value in settings.items():
            ok = name in self.supported
            if ok:
                self.values[name] = float(value)
            accepted[name] = ok
        return accepted

    def grab(self, count=3, discard=2, timeout=4.0):
        exposure = self.values["exposure"]
        bright = 135 + (20 * (exposure - IDEAL_EXPOSURE) if self.exposure_effective else 0)
        amp = 40 - abs(self.values["focus"] - IDEAL_FOCUS) / 4 if "focus" in self.supported else 30
        return [_img(bright, amp) for _ in range(count)]

    def grab_settled(self, count=3):
        return self.grab(count)

    def fps(self):
        return self.fake_fps

    def switch(self, index, width=None, height=None):
        self.width, self.height = width or self.width, height or self.height
        self.switched.append((self.width, self.height))
        return True

    def resolution(self):
        return self.width, self.height


class _Result:
    core_checks_ok = True


class FakeReader:
    """Lee la MRZ solo si el cuadro no esta ni muy oscuro ni muy claro."""

    def __init__(self, always_fail: bool = False):
        self.always_fail = always_fail

    def read(self, img):
        mean = float(img.mean())
        return None if self.always_fail or not 100 <= mean <= 170 else _Result()


def _run(camera, reader=None):
    cal = Calibrator(camera, reader or FakeReader())
    cal.state.status = "running"
    cal._run(None)
    return cal


def test_exposure_sweep_finds_the_ideal_value():
    cal = _run(FakeCamera({"exposure"}))
    assert cal.state.status == "done"
    assert cal.state.result["ajustes"]["exposure"] == IDEAL_EXPOSURE
    assert cal.state.result["validada_con_mrz"] is True
    assert cal.state.result["ganadora"] == "calibrada"  # la inicial (-6) no lee: 75 de brillo


def test_focus_sweep_finds_the_sharpest_value_and_turns_autofocus_off():
    cal = _run(FakeCamera({"exposure", "focus", "autofocus"}))
    settings = cal.state.result["ajustes"]
    assert settings["focus"] == IDEAL_FOCUS
    assert settings["autofocus"] == 0


def test_parameter_that_does_not_change_the_image_is_ignored():
    cam = FakeCamera({"exposure"}, exposure_effective=False)
    cal = _run(cam)
    caps_log = " ".join(cal.state.log)
    assert "exposure: acepta=True, efectivo=False" in caps_log
    assert cal.state.result["ajustes"]["exposure"] == -6.0  # no se toco


def test_missing_parameters_are_reported_not_invented():
    cal = _run(FakeCamera(set()))  # camara de enfoque y exposicion fijos
    assert cal.state.status == "done"
    assert cal.state.result["ajustes"] == {}


def test_without_test_document_it_warns_and_is_not_marked_validated():
    cal = _run(FakeCamera({"exposure"}), FakeReader(always_fail=True))
    assert cal.state.status == "done"
    assert cal.state.result["validada_con_mrz"] is False
    assert any("AVISO" in line for line in cal.state.log)


def test_pick_keeps_reference_unless_clearly_better():
    def cand(label, ok, sharp):
        return (label, {}, (1280, 720), {"lecturas_ok": ok, "sharp": sharp, "etiqueta": label})

    ref = cand("referencia", 3, 100.0)
    assert Calibrator._pick([cand("calibrada", 3, 105.0), ref])[0] == "referencia"  # ruido
    assert Calibrator._pick([cand("calibrada", 3, 120.0), ref])[0] == "calibrada"  # +20 %
    assert Calibrator._pick([cand("calibrada", 2, 500.0), ref])[0] == "referencia"  # lee menos
    assert Calibrator._pick([cand("1920x1080", 3, 90.0), cand("referencia", 2, 100.0)])[0] == (
        "1920x1080"
    )


def test_second_run_is_refused_while_one_is_running():
    cal = Calibrator(FakeCamera({"exposure"}), FakeReader())
    cal.state.status = "running"
    assert cal.start() is False


def test_calibration_endpoints_require_token_when_configured():
    from fastapi.testclient import TestClient

    from servicio_mrz.app import create_app
    from servicio_mrz.config import Settings

    app = create_app(Settings(camera_enabled=False, admin_token="secreto", _env_file=None))
    with TestClient(app) as c:
        assert c.get("/calibracion/estado").status_code == 401
        assert c.get("/calibracion/estado", headers={"X-Admin-Token": "mal"}).status_code == 401
        ok = c.get("/calibracion/estado", headers={"X-Admin-Token": "secreto"})
        assert ok.status_code == 200 and ok.json()["estado"] == "idle"
        assert (
            c.post("/calibracion/iniciar", headers={"X-Admin-Token": "secreto"}).status_code == 503
        )


def test_diagnostics_page_is_served():
    from fastapi.testclient import TestClient

    from servicio_mrz.app import create_app
    from servicio_mrz.config import Settings

    with TestClient(create_app(Settings(camera_enabled=False, _env_file=None))) as c:
        res = c.get("/diagnostico")
    assert res.status_code == 200 and "Iniciar calibracion" in res.text


def test_slow_exposure_is_penalized_even_if_brightness_is_ideal():
    """A exposicion larga la camara baja de 10 fps: un documento en movimiento saldria borroso."""

    class SlowAtLongExposure(FakeCamera):
        def fps(self):
            return 5.0 if self.values["exposure"] > IDEAL_EXPOSURE - 0.5 else 30.0

    cal = _run(SlowAtLongExposure({"exposure"}))
    # el ideal (-3) va a 5 fps: se elige el siguiente mas oscuro (-4) que mantiene 30 fps
    assert cal.state.result["ajustes"]["exposure"] == IDEAL_EXPOSURE - 1


# --- regresiones de los defectos hallados al revisar los logs reales ----------------------------


def test_fps_is_robust_to_the_initial_burst_of_buffered_frames():
    """Un promedio movil daba 383.281 fps tras la rafaga inicial que algunos drivers entregan."""
    from servicio_mrz.camera import Camera

    cam = Camera()
    t = 100.0
    cam._stamps.extend([t + i * 1e-6 for i in range(20)])  # rafaga: 20 cuadros en microsegundos
    cam._stamps.extend([t + 0.05 + i / 30 for i in range(11)])  # luego, 30 fps reales
    assert 25 <= cam.fps() <= 35


def test_sharpness_is_comparable_across_resolutions():
    """La nitidez de 1080p no debe parecer 'peor' que la de 720p solo por tener mas pixeles."""
    import cv2

    from servicio_mrz.calibration import frame_metrics

    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(0, 255, (360, 640), dtype=np.uint8), (0, 0), 1.0)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    hd = cv2.resize(base, (1280, 720), interpolation=cv2.INTER_CUBIC)
    fhd = cv2.resize(base, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    a, b, c = (frame_metrics(f)["sharp"] for f in (base, hd, fhd))
    assert 0.6 <= b / a <= 1.4 and 0.6 <= c / a <= 1.4, (a, b, c)


def test_two_of_three_reads_is_fragile_not_validated():
    class FlakyReader(FakeReader):
        calls = 0

        def read(self, img):
            FlakyReader.calls += 1
            return None if FlakyReader.calls % 3 == 0 else _Result()

    cal = _run(FakeCamera({"exposure"}), FlakyReader())
    result = cal.state.result
    assert result["estabilidad"] == "fragil" and result["validada_con_mrz"] is False
    assert any("FRAGIL" in line for line in cal.state.log)


def test_three_of_three_is_stable_and_zero_is_no_reading():
    assert _run(FakeCamera({"exposure"})).state.result["estabilidad"] == "estable"
    blind = _run(FakeCamera({"exposure"}), FakeReader(always_fail=True))
    assert blind.state.result["estabilidad"] == "sin_lectura"


def test_focus_is_tested_even_when_the_driver_cannot_read_it_back():
    """En DirectShow el enfoque puede leerse como -1 y aun asi aceptar cambios."""

    class UnreadableFocus(FakeCamera):
        def get(self, name):
            return -1.0 if name == "focus" else super().get(name)

    cam = UnreadableFocus({"exposure", "focus", "autofocus"})
    caps = Calibrator(cam, FakeReader()).capabilities()
    assert caps["focus"]["efectivo"] is True and caps["focus"]["acepta"] is True


def test_autofocus_is_restored_after_the_capability_probe():
    cam = FakeCamera({"exposure", "focus", "autofocus"})
    assert cam.values["autofocus"] == 1.0
    Calibrator(cam, FakeReader()).capabilities()
    assert cam.values["autofocus"] == 1.0


def test_calibration_log_is_persisted_for_later_review(tmp_path):
    cal = Calibrator(FakeCamera({"exposure"}), FakeReader(), tmp_path / "calibracion")
    cal.state.status = "running"
    cal._run(None)
    files = list((tmp_path / "calibracion").glob("calibracion_*.json"))
    assert len(files) == 1
    import json

    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["estado"] == "done" and saved["log"] and saved["resultado"]["estabilidad"]
