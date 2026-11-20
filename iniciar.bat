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

if not exist data mkdir data
rem Registro en data\servicio.log (no contiene datos de documentos). Se rota al pasar de 5 MB.
if exist data\servicio.log for %%F in (data\servicio.log) do if %%~zF GTR 5000000 move /y data\servicio.log data\servicio.anterior.log >nul

:inicio
echo [%date% %time%] arrancando el servicio >> data\servicio.log
call uv run --no-sync python -m servicio_mrz.app >> data\servicio.log 2>&1
echo [%date% %time%] el servicio se detuvo (codigo %errorlevel%); reintento en 5 s >> data\servicio.log
timeout /t 5 /nobreak >nul
goto inicio
