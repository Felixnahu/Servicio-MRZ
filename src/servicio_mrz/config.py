from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracion operativa. Los secretos viven solo en `.env` (nunca versionado)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Servicio externo (docproc / kyc-ci-backend)
    kyc_base_url: str = ""
    kyc_api_key: str = ""
    kyc_timeout_s: float = 45.0

    # Solo loopback: el stream muestra documentos de identidad.
    host: str = "127.0.0.1"
    port: int = 8765

    # Si se define, la calibracion exige la cabecera X-Admin-Token (solo tecnicos).
    admin_token: str = ""

    # Fotos de prueba guardadas por /captura/leer (gitignored; contienen datos personales)
    samples_dir: str = "samples"

    # Logs de calibracion y demas datos locales (gitignored)
    data_dir: str = "data"

    # True: la camara reconoce sola, de forma continua. False: modo foto (solo lee al pedirselo).
    auto_recognition: bool = True

    # Hilos de ONNX/OpenCV: 2 = ~1.7 s por lectura con ~2.4 nucleos; 0 = todos (mucha mas CPU).
    ocr_threads: int = 2

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_enabled: bool = True
