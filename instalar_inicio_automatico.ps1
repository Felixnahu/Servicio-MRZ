# Registra una tarea de Windows que arranca el servicio MRZ cada vez que este usuario inicia sesion.
# Ejecutar UNA vez, en el totem, desde PowerShell:
#     powershell -ExecutionPolicy Bypass -File .\instalar_inicio_automatico.ps1
# Para arrancarlo ya mismo ademas de registrarlo:  ... -File .\instalar_inicio_automatico.ps1 -IniciarAhora
param([switch]$IniciarAhora)

$ErrorActionPreference = "Stop"
$carpeta = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat = Join-Path $carpeta "iniciar.bat"
if (-not (Test-Path $bat)) { throw "No se encuentra $bat" }
if (-not (Test-Path (Join-Path $carpeta ".venv"))) {
    Write-Warning "No existe la carpeta .venv: ejecuta primero 'uv sync' en $carpeta"
}

$usuario = "$env:USERDOMAIN\$env:USERNAME"
# 'start /min' abre el servicio en una ventana minimizada; el bucle de iniciar.bat lo relanza si se cae.
$accion = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c start `"Servicio MRZ`" /min `"$bat`"" -WorkingDirectory $carpeta
$disparador = New-ScheduledTaskTrigger -AtLogOn -User $usuario
$config = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
# Interactive: la camara USB solo es accesible desde una sesion de usuario abierta.
$principal = New-ScheduledTaskPrincipal -UserId $usuario -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName "ServicioMRZ" -Action $accion -Trigger $disparador `
    -Settings $config -Principal $principal -Force | Out-Null
Write-Host "Tarea 'ServicioMRZ' registrada: arrancara al iniciar sesion $usuario."
Write-Host "Registro del servicio: $carpeta\data\servicio.log"

if ($IniciarAhora) {
    Start-ScheduledTask -TaskName "ServicioMRZ"
    Write-Host "Servicio iniciado. Abre http://127.0.0.1:8765/health para comprobarlo."
}
