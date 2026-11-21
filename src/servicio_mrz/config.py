from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracion operativa. Los secretos viven solo en `.env` (nunca versionado)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Servicio externo (docproc / kyc-ci-backend)
    kyc_base_url: str = ""
    kyc_api_key: str = ""
    kyc_timeout_s: float = 45.0
    # POST /mrz/leer: si la MRZ no se lee (cedula antigua, reflejos...) se envian las fotos a este
    # KYC para que las lea. Solo actua si KYC_BASE_URL esta configurado.
    kyc_fallback: bool = True
    # Tipos de documento (campo `documentType` de la app) que pueden ir al KYC, separados por comas.
    # Por defecto solo la cedula antigua; el KYC no lee pasaportes.
    kyc_document_types: str = "CI_BO"
    # Guarda en data/fallidas las fotos que no se pudieron leer, para medirlas despues. SOLO para
    # pruebas: son documentos reales; borrar la carpeta al terminar.
    save_failed_photos: bool = False
    # OCR local de campos impresos (~1.7 s en un PC rapido, ~20 s en el totem). En el totem va en
    # false (iniciar.bat) y las cedulas antiguas se resuelven con el KYC.
    visual_ocr: bool = True

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
    # Baja la prioridad del proceso para no quitarle CPU a otras aplicaciones (util en un equipo de
    # desarrollo). En el totem (iniciar.bat) va en false: ahi el visitante espera este resultado.
    low_priority: bool = True
    # Presupuesto blando de POST /mrz/leer: si la MRZ ya lo consumio, no se intenta el OCR de campos
    # impresos (no se puede interrumpir una vez empezado).
    request_budget_s: float = 15.0
    # Tope de POST /mrz/leer incluido el KYC. La app de SIBOSAM deja de esperar a los 45 s.
    request_deadline_s: float = 40.0

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_enabled: bool = True

    @property
    def kyc_types(self) -> set[str]:
        return {x.strip().upper() for x in self.kyc_document_types.split(",") if x.strip()}
