# Prueba aislada: OCR en CPU vs GPU (DirectML). NO toca el entorno del servicio (.venv).
# Crea un entorno aparte en la carpeta "accel_test" junto al proyecto; se puede borrar sin riesgo.
#     powershell -ExecutionPolicy Bypass -File .\bench\probar_aceleracion.ps1
$ErrorActionPreference = "Stop"
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path (Split-Path -Parent $root) "accel_test"

if (-not (Test-Path $venv)) {
    Write-Host "Creando entorno aparte en $venv ..."
    uv venv $venv --python 3.11
}
$py = Join-Path $venv "Scripts\python.exe"

Write-Host "Instalando dependencias (onnxruntime-directml en vez de onnxruntime) ..."
# rapidocr se instala sin dependencias porque exige onnxruntime (CPU), que choca con DirectML.
uv pip install --python $py onnxruntime-directml numpy opencv-python-headless pyyaml pillow `
    pyclipper shapely six tqdm
uv pip install --python $py --no-deps rapidocr-onnxruntime==1.4.4

Write-Host "Midiendo (puede tardar un par de minutos) ..."
& $py (Join-Path $PSScriptRoot "accel_bench.py")
