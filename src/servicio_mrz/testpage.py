"""Pagina local de pruebas con la camara (solo loopback). No es la interfaz del visitante."""

PAGE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tomar foto y leer</title>
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
  main { max-width:1040px; margin:0 auto; display:grid; gap:16px; }
  h1 { font-size:20px; margin:0; } h2 { font-size:15px; margin:0 0 8px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
  .grid { display:grid; gap:16px; grid-template-columns: minmax(0,1.6fr) minmax(0,1fr); }
  @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }
  .stage { position:relative; line-height:0; }
  .stage img { width:100%; border-radius:8px; background:#000; display:block; }
  .stage canvas { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
  .banner { margin-top:10px; padding:10px 14px; border-radius:8px; font-weight:600; line-height:1.4; }
  .banner.search { background:color-mix(in srgb, var(--accent) 16%, transparent); color:var(--accent); }
  .banner.done { background:color-mix(in srgb, var(--ok) 18%, transparent); color:var(--ok); }
  .banner.hint { background:color-mix(in srgb, var(--warn) 20%, transparent); color:var(--warn); }
  .banner.off { background:color-mix(in srgb, var(--muted) 18%, transparent); color:var(--muted); }
  .row { display:flex; flex-wrap:wrap; gap:10px; align-items:flex-end; }
  button { background:var(--accent); color:#fff; border:0; border-radius:8px; padding:9px 16px;
           font-size:14px; cursor:pointer; }
  button:disabled { opacity:.6; cursor:wait; }
  button.secondary { background:transparent; color:var(--accent); border:1px solid var(--accent); }
  input[type=text], select { padding:8px 10px; border:1px solid var(--line); border-radius:8px;
           background:var(--bg); color:var(--fg); font-size:14px; }
  label { display:flex; flex-direction:column; gap:3px; font-size:12px; color:var(--muted); }
  table { border-collapse:collapse; width:100%; font-size:14px; }
  td { padding:4px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
  td:first-child { color:var(--muted); width:42%; }
  pre { overflow:auto; font-size:12px; margin:8px 0 0; max-height:260px; }
  .hint { color:var(--muted); font-size:13px; }
  .ok { color:var(--ok); font-weight:600; } .no { color:var(--bad); font-weight:600; }
  code { background:var(--bg); padding:1px 5px; border-radius:4px; }
</style>
</head>
<body>
<main>
  <h1>Tomar foto y leer el documento</h1>

  <section class="card">
    <div class="row">
      <label>Camara <select id="cam"></select></label>
      <label>Resolucion
        <select id="res">
          <option value="640x480">640 x 480</option>
          <option value="1280x720" selected>1280 x 720 (HD)</option>
          <option value="1920x1080">1920 x 1080 (Full HD)</option>
        </select>
      </label>
      <button id="apply" type="button" class="secondary">Usar esta camara</button>
      <span id="camstatus" class="hint"></span>
    </div>
  </section>

  <div class="grid">
    <section class="card">
      <div class="stage"><img id="live" src="/camara/stream" alt="Vista de la camara"></div>
      <div class="row" style="margin-top:12px">
        <button id="snap" type="button" style="font-size:17px;padding:12px 26px">Tomar foto y leer</button>
        <span id="snapNote" class="hint">Pon el documento frente a la camara y pulsa el boton. El servicio no lee solo.</span>
      </div>
      <p class="hint">La vista es solo para encuadrar: no reconoce nada. La foto que se toma es la que se lee.
      Con MRZ (las lineas con <code>&lt;&lt;&lt;</code>) los datos se validan con sus digitos de control;
      sin MRZ se leen los campos impresos y el resultado queda marcado <b>para revisar</b>.</p>
    </section>

    <div style="display:grid; gap:16px; align-content:start">
      <section class="card" id="photoCard">
        <h2>Resultado</h2>
        <div id="thumbs" class="row" style="margin-bottom:10px"></div>
        <div id="photoBanner" class="banner off">Aun no se ha tomado ninguna foto.</div>
        <table id="photoFields" hidden style="margin-top:10px"></table>
        <details id="photoRawBox" hidden style="margin-top:8px"><summary>Respuesta completa (lo que recibe la app)</summary><pre id="photoRaw"></pre></details>
      </section>

      <section class="card">
        <h2>O enviar una foto que ya tengas</h2>
        <div class="row">
          <label>Foto del documento <input type="file" id="fFront" accept="image/*"></label>
          <label>Otra cara (opcional) <input type="file" id="fBack" accept="image/*"></label>
          <button id="sendPhotos" type="button" class="secondary">Enviar y leer</button>
        </div>
      </section>

      <section class="card">
        <h2>Guardar foto de prueba</h2>
        <div class="row">
          <label>Nombre <input type="text" id="name" placeholder="ej. pasaporte_bol" maxlength="40"></label>
          <button id="save" type="button" class="secondary">Guardar foto</button>
        </div>
        <p id="saved" class="hint">Guarda el cuadro actual de la camara en <code>samples/</code> (solo en esta maquina; contiene datos personales).</p>
      </section>
    </div>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const row = (k, v) => `<tr><td>${k}</td><td>${v ?? "—"}</td></tr>`;
const reloadVideo = () => { $("live").src = "/camara/stream?t=" + Date.now(); };

/* El servicio no debe reconocer solo mientras se usa esta pagina (ahorra CPU y evita lecturas dobles) */
fetch("/reconocimiento?auto=false", { method: "POST" }).catch(() => {});

/* ---------- camara ---------- */
async function loadCameras() {
  const data = await (await fetch("/camara/lista")).json();
  const sel = $("cam"); sel.innerHTML = "";
  for (const c of data.camaras) {
    const o = document.createElement("option");
    o.value = c.indice; o.textContent = c.nombre + (c.activa ? "  (en uso)" : "");
    if (c.activa) o.selected = true; sel.appendChild(o);
  }
  $("camstatus").textContent = !data.camaras.length ? "No se detectaron camaras."
    : data.resolucion ? `Activa a ${data.resolucion[0]}x${data.resolucion[1]}` : "";
}
$("apply").onclick = async () => {
  const [w, h] = $("res").value.split("x");
  $("apply").disabled = true; $("camstatus").textContent = "Cambiando de camara...";
  try {
    const res = await fetch(`/camara/elegir?indice=${$("cam").value}&ancho=${w}&alto=${h}`, { method: "POST" });
    const body = await res.json(); const d = body.detail ?? body;
    $("camstatus").textContent = res.ok ? `${d.nombre}: ${d.resolucion[0]}x${d.resolucion[1]}` : `No se pudo: ${d.mensaje ?? res.status}`;
    reloadVideo(); await loadCameras();
  } catch (e) { $("camstatus").textContent = "Error: " + e; } finally { $("apply").disabled = false; }
};

/* ---------- guardar foto de prueba ---------- */
$("save").onclick = async () => {
  $("save").disabled = true;
  try {
    const q = new URLSearchParams({ guardar: true, nombre: $("name").value });
    const res = await fetch("/captura/leer?" + q, { method: "POST" });
    const b = await res.json(); const d = b.detail ?? b;
    $("saved").textContent = d.foto ? `Guardada: samples/${d.foto}  (nitidez ${d.calidad?.nitidez}, brillo ${d.calidad?.brillo})` : "No se pudo guardar.";
  } catch (e) { $("saved").textContent = "Error: " + e; } finally { $("save").disabled = false; }
};

/* ---------- enviar la foto a /mrz/leer (el mismo flujo que usa la app) ---------- */
function photoBanner(cls, text) { const b = $("photoBanner"); b.className = "banner " + cls; b.textContent = text; }
function showThumb(src) {
  const box = $("thumbs"); box.innerHTML = "";
  const img = document.createElement("img"); img.src = src;
  img.style.cssText = "height:150px;border-radius:6px;border:1px solid var(--line)"; box.appendChild(img);
}
async function sendToMrz(front, back) {
  if (!front) { photoBanner("hint", "Elige primero una foto."); return; }
  const body = new FormData();
  body.append("file", front, front.name || "foto.jpg");
  if (back) body.append("back", back, back.name || "reverso.jpg");
  $("sendPhotos").disabled = $("snap").disabled = true;
  photoBanner("search", "Leyendo... (hasta unos 5 s si no hay MRZ y hay que leer el texto impreso)");
  $("photoFields").hidden = true;
  const t0 = performance.now();
  try {
    const res = await fetch("/mrz/leer", { method: "POST", body });
    const j = await res.json(); const secs = ((performance.now() - t0) / 1000).toFixed(1);
    $("photoRawBox").hidden = false; $("photoRaw").textContent = `HTTP ${res.status}\n` + JSON.stringify(j, null, 2);
    if (res.ok) {
      const f = j.fields, ocr = j.source === "ocr";
      photoBanner(ocr ? "hint" : "done", ocr
        ? `Leido por OCR del texto impreso en ${secs} s. SIN digitos de control: confirmar los datos con el visitante.`
        : `Leido por MRZ, controles validados, en ${secs} s`);
      $("photoFields").hidden = false;
      $("photoFields").innerHTML = row("Tipo", j.documentType) + row("Nombres", f.firstNames || "—")
        + row("Apellidos", f.lastNames || "—") + row("Numero", f.documentNumber || "—")
        + row("Nacionalidad", f.nationality || "—") + row("Nacimiento", f.birthDate)
        + row("Edad / mayor de edad", `${j.age} / ${j.isAdult}`) + row("Origen", j.source)
        + (j.reasons && j.reasons.length > 1 ? row("Avisos", j.reasons.slice(1).join(", ")) : "");
    } else {
      const d = j.detail ?? j;
      const tip = d.pista === "sin_mrz" ? " -> pedir que den la vuelta a la tarjeta" : "";
      photoBanner("hint", `No leido (HTTP ${res.status}, ${secs} s): ${d.mensaje ?? d.codigo}${tip}`);
    }
  } catch (e) { photoBanner("off", "Error: " + e); }
  finally { $("sendPhotos").disabled = $("snap").disabled = false; }
}
$("sendPhotos").onclick = () => {
  const f = $("fFront").files[0]; if (f) showThumb(URL.createObjectURL(f));
  sendToMrz(f, $("fBack").files[0]);
};

/* ---------- tomar foto: congela el cuadro que se ve y esa foto es la que se lee ---------- */
$("snap").onclick = async () => {
  $("snap").disabled = true; $("snapNote").textContent = "Tomando foto...";
  try {
    const res = await fetch("/camara/frame?t=" + Date.now());
    if (!res.ok) { $("snapNote").textContent = "La camara no entrega imagen."; return; }
    const blob = await res.blob();
    showThumb(URL.createObjectURL(blob));
    $("snapNote").textContent = "Foto tomada: esta es la que se lee.";
    await sendToMrz(new File([blob], "captura.jpg", { type: "image/jpeg" }), null);
  } catch (e) { $("snapNote").textContent = "Error: " + e; } finally { $("snap").disabled = false; }
};

loadCameras();
</script>
</body>
</html>"""
