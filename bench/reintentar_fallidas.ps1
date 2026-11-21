# Reenvia al servicio, UNA POR UNA, las fotos guardadas en data\fallidas (las mismas que mando la app)
# y muestra el resultado en una linea por intento. Sirve para saber si el fallo esta en el SERVICIO
# o en la APP: si aqui se leen, el servicio las lee bien y el problema esta en lo que manda la app
# (tipo de documento, orden de las fotos, tiempos). Si aqui NO se leen, el fallo es del servicio.
#
#     powershell -ExecutionPolicy Bypass -File .\bench\reintentar_fallidas.ps1
#     powershell -ExecutionPolicy Bypass -File .\bench\reintentar_fallidas.ps1 -Tipo PASSPORT
#     powershell -ExecutionPolicy Bypass -File .\bench\reintentar_fallidas.ps1 -Tipo CI_BO
#
# No imprime nombres, numeros ni fechas del documento: solo tiempos, origen y codigos.
param(
    [string]$Carpeta = "data\fallidas",
    [string]$Tipo = "",
    [int]$Puerto = 8765
)
$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path $Carpeta)) { throw "No existe la carpeta $Carpeta" }

$pares = Get-ChildItem $Carpeta -Filter "*_0.jpg" | Sort-Object Name
if (-not $pares) { throw "No hay fotos *_0.jpg en $Carpeta" }
$url = "http://127.0.0.1:$Puerto/mrz/leer"
Write-Host "Servicio: $url   Tipo enviado: '$Tipo'   Intentos: $($pares.Count)"
Write-Host ""

foreach ($frente in $pares) {
    $dorso = $frente.FullName -replace '_0\.jpg$', '_1.jpg'
    $curlArgs = @("-s", "-m", "120", "-X", "POST", $url, "-F", "front=@$($frente.FullName)")
    if (Test-Path $dorso) { $curlArgs += @("-F", "back=@$dorso") }
    if ($Tipo) { $curlArgs += @("-F", "documentType=$Tipo") }
    $inicio = Get-Date
    $cuerpo = & curl.exe @curlArgs
    $ms = [int]((Get-Date) - $inicio).TotalMilliseconds
    $nombre = $frente.Name -replace '_0\.jpg$', ''
    try { $r = $cuerpo | ConvertFrom-Json } catch { $r = $null }
    if ($null -eq $r) {
        Write-Host ("{0}  {1,6} ms  SIN RESPUESTA JSON (servicio apagado o tiempo agotado)" -f $nombre, $ms)
    } elseif ($r.codigo) {
        Write-Host ("{0}  {1,6} ms  NO LEIDO  codigo={2}  origen={3}  pista={4}" -f $nombre, $ms, $r.codigo, $r.origen, $r.pista)
    } else {
        $fecha = if ($r.fields.birthDate) { "con fecha de nacimiento" } else { "SIN fecha de nacimiento" }
        Write-Host ("{0}  {1,6} ms  LEIDO    fuente={2}  tipo={3}  {4}" -f $nombre, $ms, $r.source, $r.documentType, $fecha)
    }
}
