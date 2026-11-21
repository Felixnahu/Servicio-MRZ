# Baja la ultima version del repositorio, actualiza las dependencias y reinicia el servicio MRZ.
# Ejecutar en el totem, desde la carpeta del proyecto:
#     powershell -ExecutionPolicy Bypass -File .\actualizar.ps1
#
# Para probar otra rama en el totem (y volver a la estable cuando acabe):
#     powershell -ExecutionPolicy Bypass -File .\actualizar.ps1 -Rama Dev/MRZ-OCR-PASS-CI
#     powershell -ExecutionPolicy Bypass -File .\actualizar.ps1 -Rama main
param([string]$Rama = "")
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"

if (-not (Test-Path ".git")) {
    throw "Esta carpeta no es un clon de git. Ver las instrucciones de primera instalacion."
}

if ($Rama) {
    Write-Host "0/3 Cambiando a la rama $Rama..."
    git fetch origin $Rama
    if ($LASTEXITCODE -ne 0) { throw "No existe la rama $Rama en origin o no hay internet. No se toco el servicio." }
    git checkout $Rama
    if ($LASTEXITCODE -ne 0) { throw "No se pudo cambiar a $Rama (cambios locales sin guardar?). No se toco el servicio." }
}
Write-Host "Rama actual: $(git rev-parse --abbrev-ref HEAD)"

Write-Host "1/3 Bajando cambios..."
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw "git pull fallo (cambios locales sin guardar o sin internet). No se toco el servicio." }

Write-Host "2/3 Actualizando dependencias (rapido si no cambiaron)..."
uv sync
if ($LASTEXITCODE -ne 0) { throw "uv sync fallo. El servicio sigue con la version anterior." }

Write-Host "3/3 Reiniciando el servicio..."
# Termina el bucle de iniciar.bat y el servicio; se excluye este mismo proceso (su linea de comandos
# tambien contiene el patron).
Get-CimInstance Win32_Process |
    Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match 'servicio_mrz\.app|iniciar\.bat' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

if (Get-ScheduledTask -TaskName "ServicioMRZ" -ErrorAction SilentlyContinue) {
    Start-ScheduledTask -TaskName "ServicioMRZ"
    Write-Host "Servicio reiniciado por la tarea programada."
} else {
    Start-Process cmd.exe -ArgumentList "/c", "start `"Servicio MRZ`" /min `"$PSScriptRoot\iniciar.bat`""
    Write-Host "Servicio reiniciado (sin tarea programada: ejecuta instalar_inicio_automatico.ps1 para el arranque automatico)."
}
Write-Host "Version instalada: $(git log -1 --format='%h %s')"
