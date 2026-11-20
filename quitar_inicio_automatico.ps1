# Quita la tarea de inicio automatico del servicio MRZ (no borra nada del proyecto).
#     powershell -ExecutionPolicy Bypass -File .\quitar_inicio_automatico.ps1
$ErrorActionPreference = "Stop"
if (Get-ScheduledTask -TaskName "ServicioMRZ" -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName "ServicioMRZ" -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName "ServicioMRZ" -Confirm:$false
    Write-Host "Tarea 'ServicioMRZ' eliminada."
    Write-Host "Si el servicio sigue abierto, cierra su ventana (o termina cmd.exe/python.exe del servicio)."
} else {
    Write-Host "No hay ninguna tarea 'ServicioMRZ' registrada."
}
