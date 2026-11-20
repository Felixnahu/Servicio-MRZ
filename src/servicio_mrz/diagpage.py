"""Pagina de diagnostico/calibracion de la camara (solo tecnicos, solo loopback)."""

DIAG_PAGE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diagnostico de camara</title>
<style>
  :root { --bg:#f5f5f4; --fg:#1c1917; --card:#fff; --line:#d6d3d1; --ok:#15803d; --bad:#b91c1c;
          --warn:#a16207; --accent:#1d4ed8; --muted:#78716c; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#1c1917; --fg:#f5f5f4; --card:#292524; --line:#44403c; --ok:#4ade80;
            --bad:#f87171; --warn:#facc15; --accent:#60a5fa; --muted:#a8a29e; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:16px; background:var(--bg); color:var(--fg);
         font:15px/1.45 system-ui, sans-serif; }
  main { max-width:980px; margin:0 auto; display:grid; gap:16px; }
  h1 { font-size:20px; margin:0; } h2 { font-size:16px; margin:0 0 8px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
  img#live { width:100%; max-height:420px; object-fit:contain; border-radius:8px; background:#000; }
  .row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:10px; }
  button { background:var(--accent); color:#fff; border:0; border-radius:8px; padding:10px 18px;
           font-size:15px; cursor:pointer; }
  button:disabled { opacity:.55; cursor:not-allowed; }
  button.secondary { background:transparent; color:var(--accent); border:1px solid var(--accent); }
  button.danger { background:var(--bad); }
  input { padding:8px 10px; border:1px solid var(--line); border-radius:8px; background:var(--bg);
          color:var(--fg); }
  progress { width:100%; height:14px; }
  ol { margin:0; padding-left:20px; } li { margin:3px 0; }
  table { border-collapse:collapse; width:100%; margin-top:6px; font-size:14px; }
  td, th { padding:5px 8px; border-bottom:1px solid var(--line); text-align:left; }
  th { color:var(--muted); font-weight:600; }
  #log { font:12px/1.5 ui-monospace, Consolas, monospace; max-height:200px; overflow:auto;
         background:var(--bg); border-radius:8px; padding:8px; margin-top:8px; white-space:pre-wrap; }
  .badge { display:inline-block; padding:3px 10px; border-radius:99px; font-weight:600; }
  .ok { background:color-mix(in srgb, var(--ok) 18%, transparent); color:var(--ok); }
  .bad { background:color-mix(in srgb, var(--bad) 18%, transparent); color:var(--bad); }
  .warn { background:color-mix(in srgb, var(--warn) 20%, transparent); color:var(--warn); }
  .hint { color:var(--muted); font-size:13px; }
</style>
</head>
<body>
<main>
  <h1>Diagnostico y calibracion de la camara</h1>

  <section class="card">
    <h2>Antes de empezar</h2>
    <ol>
      <li>Elige la camara en <a href="/probar">/probar</a> (esta pagina calibra la que este en uso).</li>
      <li>Coloca un documento de prueba <b>con MRZ</b> en un soporte, <b>quieto</b>, en la posicion
          real de uso del totem, con la iluminacion real.</li>
      <li>Pulsa <b>Iniciar calibracion</b> y no muevas nada hasta que termine (~1 minuto).</li>
    </ol>
    <p class="hint">Solo hace falta al instalar o cambiar el hardware. Sin documento de prueba la
    calibracion no puede comprobar que la MRZ se lee y lo avisa.</p>
    <div class="row">
      <label class="hint">Token tecnico (si esta configurado)
        <input type="password" id="token" autocomplete="off"></label>
    </div>
  </section>

  <section class="card">
    <img id="live" src="/camara/stream" alt="Video en vivo de la camara">
    <div class="row">
      <button id="caps" class="secondary">Ver capacidades</button>
      <button id="start">Iniciar calibracion</button>
      <button id="cancel" class="danger" disabled>Cancelar</button>
      <button id="reset" class="secondary">Restablecer</button>
    </div>
  </section>

  <section class="card" id="capsCard" hidden>
    <h2>Que controla esta camara</h2>
    <table id="capsTable"></table>
    <p class="hint">"Efectivo" = el brillo o la nitidez cambian de verdad al mover el parametro.
    Un parametro que el driver acepta pero no surte efecto se ignora.</p>
  </section>

  <section class="card" id="runCard" hidden>
    <h2 id="phase">Calibrando</h2>
    <progress id="bar" max="100" value="0"></progress>
    <div id="log"></div>
  </section>

  <section class="card" id="resultCard" hidden>
    <h2>Resultado</h2>
    <div id="verdict"></div>
    <table id="settings"></table>
    <h2 style="margin-top:14px">Configuraciones probadas</h2>
    <table id="cands"></table>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const headers = () => { const t = $("token").value; return t ? { "X-Admin-Token": t } : {}; };
try { $("token").value = sessionStorage.getItem("tok") || ""; } catch (e) {}
$("token").oninput = () => { try { sessionStorage.setItem("tok", $("token").value); } catch (e) {} };
let timer = null;

async function api(path, method = "GET") {
  const res = await fetch(path, { method, headers: headers() });
  const body = await res.json();
  if (!res.ok) { const d = body.detail ?? body; throw new Error(d.mensaje ?? res.status); }
  return body;
}
const yn = (b) => b === null ? '<span class="hint">no evaluado</span>'
  : b ? '<span class="badge ok">si</span>' : '<span class="badge warn">no</span>';

$("caps").onclick = async () => {
  $("caps").disabled = true;
  try {
    const c = await api("/calibracion/capacidades");
    $("capsCard").hidden = false;
    $("capsTable").innerHTML = "<tr><th>Parametro</th><th>Valor actual</th><th>Acepta</th><th>Efectivo</th></tr>"
      + Object.entries(c).map(([k, v]) =>
        `<tr><td>${k}</td><td>${v.actual ?? "—"}</td><td>${yn(v.acepta)}</td><td>${yn(v.efectivo)}</td></tr>`).join("");
  } catch (e) { alert(e.message); } finally { $("caps").disabled = false; }
};

$("start").onclick = async () => {
  try { await api("/calibracion/iniciar", "POST"); } catch (e) { alert(e.message); return; }
  $("resultCard").hidden = true; $("runCard").hidden = false; poll();
};
$("cancel").onclick = () => api("/calibracion/cancelar", "POST").catch((e) => alert(e.message));
$("reset").onclick = async () => {
  if (!confirm("Borrar la calibracion guardada y reabrir la camara?")) return;
  try { const r = await api("/calibracion/restablecer", "POST");
        alert(r.restablecida ? "Calibracion borrada." : "No habia calibracion guardada.");
        $("live").src = "/camara/stream?t=" + Date.now();
  } catch (e) { alert(e.message); }
};

function poll() {
  clearTimeout(timer);
  api("/calibracion/estado").then((s) => {
    const running = s.estado === "running";
    $("start").disabled = running; $("cancel").disabled = !running;
    $("phase").textContent = running ? "Calibrando: " + s.fase : "Estado: " + s.estado;
    $("bar").value = s.progreso;
    $("log").textContent = s.log.join("\\n"); $("log").scrollTop = 1e9;
    if (running) { timer = setTimeout(poll, 1000); return; }
    if (s.estado === "done" && s.resultado) showResult(s.resultado);
    $("live").src = "/camara/stream?t=" + Date.now();
  }).catch((e) => { $("log").textContent = e.message; });
}

function showResult(r) {
  $("resultCard").hidden = false;
  const ev = r.evaluacion;
  $("verdict").innerHTML = r.estabilidad === "estable"
    ? `<span class="badge ok">Estable: la MRZ se leyo ${ev.lecturas_ok}/${ev.lecturas} veces</span>`
    : r.estabilidad === "fragil"
    ? `<span class="badge warn">FRAGIL: solo ${ev.lecturas_ok}/${ev.lecturas} lecturas</span> <span class="hint">Una captura suelta falla a menudo. Mejora la luz, la distancia o el enfoque y repite.</span>`
    : `<span class="badge bad">NO se leyo la MRZ</span> <span class="hint">Repite con un documento de prueba con MRZ quieto frente a la camara.</span>`;
  const s = r.ajustes;
  $("settings").innerHTML = `<tr><th>Ganadora</th><td>${r.ganadora}</td></tr>`
    + `<tr><th>Resolucion</th><td>${r.resolucion[0]} x ${r.resolucion[1]}</td></tr>`
    + Object.entries(s).map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("")
    + `<tr><th>Guardada</th><td>Si, se aplicara al arrancar el servicio</td></tr>`;
  $("cands").innerHTML = "<tr><th>Configuracion</th><th>Lecturas OK</th><th>ms/lectura</th><th>Nitidez</th><th>Brillo</th><th>Reflejo %</th><th>FPS</th></tr>"
    + r.candidatas.map((c) => `<tr><td>${c.etiqueta}</td><td>${c.lecturas_ok}/${c.lecturas}</td><td>${c.ms_medio}</td><td>${c.sharp}</td><td>${c.bright}</td><td>${c.glare}</td><td>${c.fps}</td></tr>`).join("");
}
poll();
</script>
</body>
</html>
"""
