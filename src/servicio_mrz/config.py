from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracion operativa. Los secretos viven solo en `.env` (nunca versionado)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Guarda en data/fallidas las fotos que no se pudieron leer, para medirlas despues. SOLO para
    # pruebas: son documentos reales; borrar la carpeta al terminar.
    save_failed_photos: bool = False
    # OCR local de campos impresos (~1.7 s en un PC rapido, ~20 s en el totem). En el totem va en
    # false (iniciar.bat): las cedulas antiguas se leen con la plantilla de `old_ci_ocr`.
    visual_ocr: bool = True
    # Cedula antigua (sin MRZ): lectura local por plantilla (~0.8 s por foto en un PC rapido). Una
    # observacion MRZ falsa no impide probar esta plantilla.
    old_ci_ocr: bool = True
    # Tope (segundos desde que llega la peticion) del OCR del carnet antiguo.
    old_ci_deadline_s: float = 28.0
    # Cedula 2023: tras leer la MRZ, lee el "lugar de nacimiento" del reverso (franja superior,
    # ~0.3 s) para `isSamaipateno`. Si falla o no hay tiempo, queda en null.
    birthplace_ocr: bool = True
    # Lugar de nacimiento que cuenta como samaipateno. Solo se cambia para PROBAR con un carnet de
    # otro lugar (p. ej. BIRTHPLACE_TARGET=TUPIZA); en produccion, el valor por defecto.
    birthplace_target: str = "SAMAIPATA"

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
    # Tiempo que la lectura de MRZ puede seguir probando variantes y alineaciones tras el primer
    # intento. Es tiempo ABSOLUTO: en un equipo ~5x mas lento (el totem) 2.7 s alcanzan para una
    # sola variante y se pierden lecturas que el mismo codigo logra en un PC rapido.
    ocr_budget_s: float = 2.7
    # Arena de memoria de ONNX: ~1.75x mas rapido con fotos nuevas (formas de entrada cambiantes) a
    # cambio de ~1 GB de RAM (se estabiliza en ~1.3 GB en total). false = ~0.45 GB y mas lento.
    ocr_mem_arena: bool = True
    # Baja la prioridad del proceso para no quitarle CPU a otras aplicaciones (util en un equipo de
    # desarrollo). En el totem (iniciar.bat) va en false: ahi el visitante espera este resultado.
    low_priority: bool = True
    # Prioridad explicita del proceso: below_normal, normal o above_normal (vacio = la que dicte
    # low_priority). En el totem iniciar.bat usa above_normal: la tarea programada de Windows lo
    # arranca por DEBAJO de lo normal y la app le quita la CPU.
    priority: str = ""
    # Presupuesto blando de POST /mrz/leer: si la MRZ ya lo consumio, no se intenta el OCR de campos
    # impresos (no se puede interrumpir una vez empezado).
    request_budget_s: float = 15.0
    # Tope de POST /mrz/leer. La app de SIBOSAM deja de esperar a los 45 s.
    request_deadline_s: float = 40.0

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_enabled: bool = True
