@echo off
rem Arranca el servicio MRZ y lo relanza si se detiene. Lo usa la tarea programada de Windows
rem (instalar_inicio_automatico.ps1), pero tambien se puede ejecutar a mano.
rem Requiere haber ejecutado antes `uv sync` una vez en esta carpeta.

cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
title Servicio MRZ

rem En el totem la APP toma las fotos con su propia camara y las manda al servicio: el servicio NO
rem debe abrirla (una webcam USB suele dejarse abrir por un solo programa). Para que el servicio
rem use la camara (pagina /probar), definir CAMERA_ENABLED=true antes de ejecutar este script.
if not defined CAMERA_ENABLED set "CAMERA_ENABLED=false"
rem En el totem el visitante espera la lectura: prioridad normal (en la laptop de desarrollo baja).
if not defined LOW_PRIORITY set "LOW_PRIORITY=false"
rem La tarea programada arranca los procesos con prioridad por DEBAJO de lo normal y la app del
rem totem (pantalla + camara) le quita la CPU al OCR. Subirla no pide administrador.
if not defined PRIORITY set "PRIORITY=above_normal"
rem El OCR local de campos impresos tarda ~20 s en el totem: las cedulas antiguas (sin MRZ) se leen
rem con la plantilla local (OLD_CI_OCR). Para volver a usar el OCR local: definir VISUAL_OCR=true antes de este script.
if not defined VISUAL_OCR set "VISUAL_OCR=false"
rem El totem es ~5x mas lento que un PC de desarrollo: con el presupuesto por defecto (2.7 s) la lectura
rem de MRZ se corta tras una sola variante y pierde pasaportes que si se leen con mas tiempo.
rem Una variable de entorno gana sobre .env, asi que solo se fija si .env no la define (para poder
rem probar otros valores desde .env sin editar este archivo).
if not defined OCR_BUDGET_S (findstr /b /i /c:"OCR_BUDGET_S=" .env >nul 2>&1 || set "OCR_BUDGET_S=12")

if not exist data mkdir data
rem Registro en data\servicio.log (no contiene datos de documentos). Se rota al pasar de 5 MB.
if exist data\servicio.log for %%F in (data\servicio.log) do if %%~zF GTR 5000000 move /y data\servicio.log data\servicio.anterior.log >nul

:inicio
echo [%date% %time%] arrancando el servicio >> data\servicio.log
call uv run --no-sync python -m servicio_mrz.app >> data\servicio.log 2>&1
echo [%date% %time%] el servicio se detuvo (codigo %errorlevel%); reintento en 5 s >> data\servicio.log
timeout /t 5 /nobreak >nul
goto inicio
