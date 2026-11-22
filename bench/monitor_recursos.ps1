# Registra cada pocos segundos cuanta CPU y RAM consume el totem y QUIEN la consume (servicio, app,
# AnyDesk, Defender...). Pensado para dejarlo corriendo mientras se usa la app SIBOSAM con el servicio.
#
#     powershell -ExecutionPolicy Bypass -File .\bench\monitor_recursos.ps1
#     powershell -ExecutionPolicy Bypass -File .\bench\monitor_recursos.ps1 -Minutos 10 -Cada 1
#
# Para parar antes: Ctrl+C (el resumen se imprime igual). Escribe data\recursos.csv (hora local, la
# misma que data\servicio.log, para cruzar cada escaneo con la carga). Solo nombres de procesos y
# porcentajes: no guarda datos de documentos. No usa los contadores de rendimiento de Windows (en este
# totem estan danados): calcula la CPU con el tiempo de procesador de cada proceso.
param(
    [double]$Minutos = 15,
    [double]$Cada = 2,
    [string]$Salida = "data\recursos.csv"
)
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
$nucleos = [Environment]::ProcessorCount
$grupos = [ordered]@{
    servicio = { param($p, $svc) $svc -contains $p.Id }
    app      = { param($p, $svc) $p.ProcessName -like "SIBOSAM*" -or $p.ProcessName -like "msedgewebview2*" }
    anydesk  = { param($p, $svc) $p.ProcessName -like "AnyDesk*" }
    defender = { param($p, $svc) $p.ProcessName -in @("MsMpEng", "MpDefenderCoreService", "NisSrv") }
    windows  = { param($p, $svc) $p.ProcessName -in @("explorer", "dwm", "SearchHost", "StartMenuExperienceHost", "TextInputHost", "sihost", "ctfmon") }
}

function Get-Cpu {
    $t = @{}
    foreach ($p in Get-Process) {
        try { $t[$p.Id] = @($p.TotalProcessorTime.TotalSeconds, $p.ProcessName) } catch { }
    }
    @{ hora = Get-Date; procesos = $t }
}

$dir = Split-Path -Parent $Salida
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory $dir | Out-Null }
"hora,cpu_total_pct,ram_libre_mb,servicio_pct,app_pct,anydesk_pct,defender_pct,windows_pct,otros_pct,top3" |
    Out-File $Salida -Encoding ascii
Write-Host "Midiendo cada $Cada s durante $Minutos min -> $Salida  (Ctrl+C para parar)"
Write-Host "nucleos: $nucleos. Los porcentajes son del equipo completo (100 = los 4 nucleos al maximo)."

$filas = @()
$ant = Get-Cpu
$fin = (Get-Date).AddMinutes($Minutos)
try {
    while ((Get-Date) -lt $fin) {
        Start-Sleep -Milliseconds ([int]($Cada * 1000))
        $act = Get-Cpu
        # el intervalo real es entre las dos FOTOS de procesos (con el equipo saturado el muestreo
        # tarda mas de lo previsto; dividir por el tiempo teorico inflaba los porcentajes)
        $dt = ($act.hora - $ant.hora).TotalSeconds
        $actP = $act.procesos; $antP = $ant.procesos
        # procesos de python que ejecutan el servicio (y el lanzador uv que los arranca)
        $svc = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='uv.exe'" |
            Where-Object { $_.CommandLine -like "*servicio_mrz*" } | ForEach-Object { $_.ProcessId })
        $suma = @{}; foreach ($g in $grupos.Keys) { $suma[$g] = 0.0 }
        $suma["otros"] = 0.0
        $porProceso = @{}
        foreach ($id in $actP.Keys) {
            if (-not $antP.ContainsKey($id)) { continue }
            $d = ($actP[$id][0] - $antP[$id][0]) / ($dt * $nucleos) * 100
            if ($d -lt 0) { continue }
            $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
            if (-not $proc) { continue }
            $grupo = "otros"
            foreach ($g in $grupos.Keys) { if (& $grupos[$g] $proc $svc) { $grupo = $g; break } }
            $suma[$grupo] += $d
            $porProceso[$actP[$id][1]] = [double]$porProceso[$actP[$id][1]] + $d
        }
        $total = ($suma.Values | Measure-Object -Sum).Sum
        $libre = [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB)
        $top = ($porProceso.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 3 |
            ForEach-Object { "{0}={1:N0}" -f $_.Key, $_.Value }) -join ";"
        $f = [pscustomobject]@{
            hora = (Get-Date).ToString("HH:mm:ss"); total = $total; libre = $libre
            servicio = $suma["servicio"]; app = $suma["app"]; anydesk = $suma["anydesk"]
            defender = $suma["defender"]; windows = $suma["windows"]; otros = $suma["otros"]; top = $top
        }
        $filas += $f
        ("{0},{1:N0},{2},{3:N0},{4:N0},{5:N0},{6:N0},{7:N0},{8:N0},{9}" -f $f.hora, $f.total, $f.libre,
            $f.servicio, $f.app, $f.anydesk, $f.defender, $f.windows, $f.otros, $f.top) |
            Out-File $Salida -Append -Encoding ascii
        Write-Host ("{0}  CPU {1,3:N0}%  servicio {2,3:N0}  app {3,3:N0}  anydesk {4,3:N0}  defender {5,3:N0}  RAM libre {6} MB" -f
            $f.hora, $f.total, $f.servicio, $f.app, $f.anydesk, $f.defender, $f.libre)
        if ($dt -gt ($Cada * 2)) { Write-Host ("   (intervalo real {0:N0} s: el equipo estaba saturado)" -f $dt) }
        $ant = $act
    }
} finally {
    if ($filas.Count) {
        Write-Host ""
        Write-Host "=== RESUMEN ($($filas.Count) muestras) ==="
        foreach ($c in "total", "servicio", "app", "anydesk", "defender", "windows", "otros") {
            $m = $filas | Measure-Object $c -Average -Maximum
            Write-Host ("{0,-9} media {1,5:N1}%   maximo {2,5:N1}%" -f $c, $m.Average, $m.Maximum)
        }
        $l = $filas | Measure-Object libre -Minimum
        Write-Host ("RAM libre minima: {0} MB" -f $l.Minimum)
        $calientes = @($filas | Where-Object { $_.total -ge 85 }).Count
        Write-Host ("Muestras con la CPU al 85% o mas: {0} de {1}" -f $calientes, $filas.Count)
        Write-Host "Detalle por muestra en $Salida"
    }
}
