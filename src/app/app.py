import json
from pathlib import Path
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

DATA_CSV = Path("src/data/temperature/temp_per_country/yearly_temp_aggregated/country_year.csv")
ANOM_CLIP = (-3.0, 3.0)

BLOG_HTML = """
<h1 style="margin: 0 0 8px 0;">What is EmissionWiz?</h1>
<p style="margin: 0 0 12px 0;">
  EmissionWiz is an interactive globe that lets you explore how countries have warmed over time.
  It visualizes either <b>temperature anomalies</b> (change relative to a 1991–2020 baseline)
  or <b>absolute annual temperatures</b>.
</p>
<h2 style="margin: 16px 0 6px 0;">How to read the colors</h2>
<ul style="margin: 0 0 12px 18px;">
  <li><b>Blue → White → Red</b>: cooler to warmer along the selected scale.</li>
  <li>
    In <b>Anomaly</b> mode, red means the selected year is warmer than that country’s
    1991–2020 average; blue means cooler.
  </li>
  <li>In <b>Absolute</b> mode, colors map to actual °C (cold to hot climates).</li>
</ul>
<h2 style="margin: 16px 0 6px 0;">Two metrics, two stories</h2>
<ul style="margin: 0 0 12px 18px;">
  <li><b>Anomaly (ΔT)</b>: best for seeing <i>change</i> within each country over time.</li>
  <li><b>Absolute (°C)</b>: best for communicating the <i>climate people experience</i> (intuitive values in °C).</li>
</ul>
<h2 style="margin: 16px 0 6px 0;">How to use it</h2>
<ol style="margin: 0 0 12px 18px;">
  <li>Pick <b>Anomaly</b> or <b>Absolute</b> at the top-right.</li>
  <li>Drag the <b>year slider</b> to travel through time.</li>
  <li><b>Hover</b> a country to see its value for the selected year.</li>
  <li>Open the <b>Guide</b> (top-left) for context and notes during your exploration.</li>
</ol>
<h2 style="margin: 16px 0 6px 0;">Interpreting the data</h2>
<ul style="margin: 0 0 12px 18px;">
  <li><b>Long-term warming</b> appears as a shift toward reds in anomaly mode across successive years.</li>
  <li>
    <b>Year-to-year wiggles</b> reflect natural variability; the trend over decades tells the climate story.
  </li>
  <li><b>Regional contrasts</b> highlight uneven warming—e.g., high latitudes often warm faster.</li>
</ul>
<h2 style="margin: 16px 0 6px 0;">Methodology (short)</h2>
<ul style="margin: 0 0 12px 18px;">
  <li>Source: CRU TS v4.09 (country-aggregated annual means).</li>
  <li>Anomalies: year minus each country’s 1991–2020 mean.</li>
  <li>Aggregation: monthly to annual means; countries require sufficient monthly coverage.</li>
  <li>Country names are harmonized; small territories may be excluded.</li>
</ul>
<h2 style="margin: 16px 0 6px 0;">Limitations</h2>
<ul style="margin: 0 0 12px 18px;">
  <li>Not all territories have complete records; some small islands or disputed regions may be missing.</li>
  <li>Country averages hide sub-national extremes; local conditions can differ.</li>
  <li>Absolute °C values depend on elevation, latitude, and observational coverage.</li>
</ul>
<h2 style="margin: 16px 0 6px 0;">Things to explore</h2>
<ul style="margin: 0 0 12px 18px;">
  <li>Compare early 20th century vs. recent decades in anomaly mode.</li>
  <li>Identify the fastest-warming regions and discuss likely drivers.</li>
  <li>Switch to absolute °C to relate climate zones to lived experience.</li>
</ul>
<p style="margin: 16px 0 0 0; color: #bbb; font-size: 12px;">
  Tip: Use anomaly mode for trend detection; use absolute °C for intuitive communication.
</p>
"""
BLOG_JSON = json.dumps(BLOG_HTML)

st.set_page_config(page_title="EmissionWiz", page_icon="🌍", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root, html, body { margin:0; padding:0; height:100%; overflow:hidden; background:#000; }
#MainMenu, header, footer { display:none !important; }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="collapsedControl"] { display:none !important; }
[data-testid="stAppViewContainer"] { padding:0 !important; overflow:hidden !important; }
.block-container { padding:0 !important; margin:0 !important; max-width:100% !important; }
[data-testid="stIFrame"]        { position:fixed !important; inset:0 !important; width:100vw !important; height:100vh !important; border:none !important; border-radius:0 !important; box-shadow:none !important; }
[data-testid="stIFrame"] iframe { width:100vw !important; height:100vh !important; display:block; }
.info-panel{
  position: fixed; left:16px; top:50%; transform:translateY(-50%);
  z-index: 9999; width: 280px;
  background: rgba(0,0,0,.60); color:#fff; border:1px solid rgba(255,255,255,.25);
  border-radius:12px; padding:12px 12px 10px 12px; backdrop-filter: blur(4px);
  display:none;
}
.info-panel.show{ display:block; }
.info-head{ display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
.info-title{ font-weight:700; letter-spacing:.3px; }
.info-close{ background:rgba(255,255,255,.12); color:#fff; border:1px solid rgba(255,255,255,.25);
  border-radius:8px; padding:4px 8px; cursor:pointer; }
.info-text{ font:12px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; color:#ddd; margin-bottom:8px; white-space:normal;}
.info-svg{ width:100%; height:90px; display:block; border:1px solid rgba(255,255,255,.15); border-radius:8px; background:rgba(255,255,255,.05); }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_payload(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    req = {"country", "year", "temp_c", "base", "anom"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    df["country_norm"] = (df["country"].astype(str).str.replace("_", " ", regex=False).str.strip())
    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    years_str = [str(y) for y in years]
    values_anom, values_abs = {}, {}
    for y in years:
        sub = df[df["year"] == y]
        values_anom[str(y)] = {c: float(v) for c, v in zip(sub["country_norm"], sub["anom"].round(3))}
        values_abs[str(y)] = {c: float(v) for c, v in zip(sub["country_norm"], sub["temp_c"].round(2))}
    q1, q99 = df["temp_c"].quantile([0.01, 0.99]).tolist()
    abs_clip = (float(round(q1, 1)), float(round(q99, 1)))
    return {
        "years": years_str,
        "values": {"anom": values_anom, "abs": values_abs},
        "clips": {"anom": ANOM_CLIP, "abs": abs_clip},
        "units": {"anom": "Relative Temperature Deviation ΔT (°C)", "abs": "Temperature (°C)"},
        "default_metric": "anom"
    }

payload = load_payload(DATA_CSV)
PAYLOAD_JSON = json.dumps(payload)

HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  html,body{margin:0; padding:0; height:100vh; background:#000; overflow:hidden}
  #root{position:fixed; inset:0; background:#000;}
  .panel{
    position: fixed; top:16px; right:16px; z-index: 9998;
    background: rgba(0,0,0,.45); color:#fff; padding:10px 12px; border-radius:10px;
    font: 12px/1.35 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
    border:1px solid rgba(255,255,255,.2); backdrop-filter: blur(3px);
    max-width: 600px;
  }
  .panel .row{display:flex; gap:8px; align-items:center; margin-bottom:12px; flex-wrap:nowrap}
  .panel button{background:rgba(255,255,255,.12); color:#fff; border:1px solid rgba(255,255,255,.25); border-radius:8px; padding:6px 10px; cursor:pointer}
  .panel button.active{background:#fff; color:#000}
  .grad{width:220px; height:10px; margin:6px 0 4px;}
  .scale{width:220px; display:flex; justify-content:space-between}
  #range{width:220px;}
  #sel{margin-top:4px; font-weight:600;}
  .blog-btn{
    position:fixed; top:16px; left:16px; z-index:9999;
    background:rgba(255,255,255,.12); color:#fff; border:1px solid rgba(255,255,255,.25);
    border-radius:10px; padding:8px 12px; cursor:pointer; backdrop-filter: blur(3px);
  }
  .blog-overlay{
    position:fixed; inset:0; z-index:10000; display:none; background:rgba(0,0,0,.85);
    animation: slideDown .25s ease-out;
  }
  .blog-overlay.show{display:block;}
  @keyframes slideDown{from{transform:translateY(-10%); opacity:.0} to{transform:translateY(0); opacity:1}}
  .blog-wrap{position:absolute; top:0; left:0; right:0; bottom:0; display:flex; flex-direction:column; gap:12px; padding:20px;}
  .blog-bar{display:flex; align-items:center; justify-content:space-between;}
  .blog-title{color:#fff; font-weight:700; letter-spacing:.3px;}
  .blog-actions button{
    background:rgba(255,255,255,.12); color:#fff; border:1px solid rgba(255,255,255,.25);
    border-radius:8px; padding:6px 10px; cursor:pointer; margin-left:8px;
  }
  .blog-area{
    flex:1; width:100%; border-radius:12px; border:1px solid rgba(255,255,255,.2);
    background:rgba(0,0,0,.25); color:#fff; padding:14px;
  }
  .info-panel{
    position: fixed; left:16px; top:50%; transform:translateY(-50%);
    z-index: 9999; width: 280px;
    background: rgba(0,0,0,.60); color:#fff; border:1px solid rgba(255,255,255,.25);
    border-radius:12px; padding:12px 12px 10px 12px; backdrop-filter: blur(4px);
    display:none;
  }
  .info-panel.show{ display:block; }
  .info-head{ display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
  .info-title{ font-weight:700; letter-spacing:.3px; }
  .info-close{ background:rgba(255,255,255,.12); color:#fff; border:1px solid rgba(255,255,255,.25);
    border-radius:8px; padding:4px 8px; cursor:pointer; }
  .info-text{ font:12px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; color:#ddd; margin-bottom:8px; white-space:normal;}
  .info-svg{ width:100%; height:90px; display:block; border:1px solid rgba(255,255,255,.15); border-radius:8px; background:rgba(255,255,255,.05); }
</style>
<script src="https://unpkg.com/three@0.155.0/build/three.min.js"></script>
<script src="https://unpkg.com/globe.gl@2.33.1/dist/globe.gl.min.js"></script>
</head>
<body>
<div id="root"></div>

<div class="info-panel" id="info">
  <div class="info-head">
    <div class="info-title" id="infoTitle">Country</div>
    <button class="info-close" id="infoClose">Close</button>
  </div>
  <div class="info-text" id="infoText"></div>
  <svg class="info-svg" id="infoSvg" viewBox="0 0 260 90" preserveAspectRatio="none"></svg>
</div>

<button class="blog-btn" id="openBlog">Guide</button>
<div class="blog-overlay" id="blog">
  <div class="blog-wrap">
    <div class="blog-bar">
      <div class="blog-title">EmissionWiz</div>
      <div class="blog-actions">
        <button id="blogClose">Close</button>
      </div>
    </div>
    <div id="blogContent" class="blog-area" style="overflow:auto; white-space:normal; font:14px/1.55 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;"></div>
  </div>
</div>

<div class="panel">
  <div class="row">
    <button id="btn-anom">Anomaly</button>
    <button id="btn-abs">Absolute</button>
    <button id="btn-cb">Colorblind: OFF</button>
    <button id="btn-png">Export PNG</button>
  </div>
  <div><b id="unit">__UNIT__</b></div>
  <div class="grad" id="gradBar"></div>
  <div class="scale"><span id="minlbl">__MIN__</span><span id="maxlbl">__MAX__</span></div>
  <input id="range" type="range" min="0" max="__MAXIDX__" value="__MAXIDX__" />
  <div id="sel"></div>
</div>

<script>
  const PAYLOAD = __PAYLOAD__;
  const YEARS   = PAYLOAD.years;
  const VALUES  = PAYLOAD.values;
  const CLIPS   = PAYLOAD.clips;
  const UNITS   = PAYLOAD.units;
  const BLOG    = __BLOG__;

  const ALIASES = {
    "United States of America": "USA",
    "W. Sahara": "Western Sahara",
    "Dem. Rep. Congo": "DR Congo",
    "Dominican Rep.": "Dominican Republic",
    "Falkland Is.": "Falkland Isl",
    "Fr. S. Antarctic Lands": "French Southern Territories",
    "Timor-Leste": "East Timor",
    "Côte d'Ivoire": "Ivory Coast",
    "Central African Rep.": "Central African Rep",
    "Eq. Guinea": "Equatorial Guinea",
    "eSwatini": "Swaziland",
    "Vanuatu": "Vanatu",
    "Solomon Is.": "Solomon Isl",
    "Czechia": "Czech Republic",
    "Bosnia and Herz.": "Bosnia-Herzegovinia",
    "North Macedonia": "Macedonia",
    "S. Sudan": "South Sudan",
    "Antarctica": null, "N. Cyprus": null, "Somaliland": null,
    "French Southern Territories": null,
    "Puerto Rico":"Puerto Rico", "Taiwan":"Taiwan"
  };

  let selectedCountry = null;
  let playing = false;
  let playTimer = null;
  let scheme = 'normal';

  function csvName(neName) {
    const raw = String(neName || "").trim();
    const a = Object.prototype.hasOwnProperty.call(ALIASES, raw) ? ALIASES[raw] : raw;
    if (a === null) return null;
    return a.replaceAll("_"," ").replaceAll(".","").trim();
  }
  function getValue(map, key){
    if (!key) return null;
    if (key in map) return map[key];
    const space = key.replaceAll("-", " ");
    if (space in map) return map[space];
    const hyph  = key.replaceAll(" ", "-");
    if (hyph in map) return map[hyph];
    return null;
  }
  function seriesForCountry(name, metricKey){
    const key = csvName(name);
    const alt1 = key?.replaceAll('-', ' ');
    const alt2 = key?.replaceAll(' ', '-');
    const ys = [];
    for(let i=0;i<YEARS.length;i++){
      const y = YEARS[i];
      const map = VALUES[metricKey][y] || {};
      const v = (key && key in map) ? map[key]
            : (alt1 && alt1 in map) ? map[alt1]
            : (alt2 && alt2 in map) ? map[alt2]
            : null;
      ys.push(v);
    }
    return ys;
  }
  function linreg(yvals){
    const x = []; const y = [];
    for(let i=0;i<yvals.length;i++){
      if (yvals[i] != null && !isNaN(yvals[i])) { x.push(i); y.push(yvals[i]); }
    }
    if (x.length < 2) return {slope:0, intercept: y[y.length-1] ?? 0};
    const n = x.length;
    const sx = x.reduce((a,b)=>a+b,0), sy = y.reduce((a,b)=>a+b,0);
    const sxx = x.reduce((a,b)=>a+b*b,0), sxy = x.reduce((a,b,i)=>a+b*y[i],0);
    const denom = n*sxx - sx*sx || 1e-9;
    const slope = (n*sxy - sx*sy) / denom;
    const intercept = (sy - slope*sx)/n;
    return {slope, intercept};
  }
  function sparklineSVG(svgEl, data, opts){
    const W=260, H=90, PADL=38, PADR=8, PADT=10, PADB=24;
    const xTicks = opts?.xTicks ?? [];
    const yTicks = opts?.yTicks ?? [];
    const yUnit  = opts?.yUnit  ?? '';
    svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svgEl.innerHTML = '';
    const valid = data.filter(v=>v!=null && !isNaN(v));
    if (valid.length<2){ return; }
    const min = Math.min(...valid), max = Math.max(...valid);
    const rng = (max-min)||1e-6;
    const sx = (i)=> PADL + (W-PADL-PADR)*i/(data.length-1||1);
    const sy = (v)=> H-PADB - (H-PADT-PADB)*((v-min)/rng);
    const ns = 'http://www.w3.org/2000/svg';
    const gAxis = document.createElementNS(ns,'g');
    gAxis.setAttribute('stroke','rgba(255,255,255,0.35)');
    gAxis.setAttribute('stroke-width','1');
    const x0 = PADL, x1 = W-PADR, y0 = H-PADB, y1 = PADT;
    const xAxis = document.createElementNS(ns,'line');
    xAxis.setAttribute('x1', x0); xAxis.setAttribute('y1', y0);
    xAxis.setAttribute('x2', x1); xAxis.setAttribute('y2', y0);
    gAxis.appendChild(xAxis);
    const yAxis = document.createElementNS(ns,'line');
    yAxis.setAttribute('x1', x0); yAxis.setAttribute('y1', y0);
    yAxis.setAttribute('x2', x0); yAxis.setAttribute('y2', y1);
    gAxis.appendChild(yAxis);
    const gGrid = document.createElementNS(ns,'g');
    gGrid.setAttribute('stroke','rgba(255,255,255,0.15)');
    gGrid.setAttribute('stroke-width','1');
    const gLab = document.createElementNS(ns,'g');
    gLab.setAttribute('fill','#ddd');
    gLab.setAttribute('font-size','9');
    gLab.setAttribute('font-family','system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif');
    xTicks.forEach(t=>{
      const idx = YEARS.indexOf(String(t));
      if (idx<0) return;
      const X = sx(idx);
      const tick = document.createElementNS(ns,'line');
      tick.setAttribute('x1', X); tick.setAttribute('y1', y0);
      tick.setAttribute('x2', X); tick.setAttribute('y2', y0+4);
      gAxis.appendChild(tick);
      const grid = document.createElementNS(ns,'line');
      grid.setAttribute('x1', X); grid.setAttribute('y1', y0);
      grid.setAttribute('x2', X); grid.setAttribute('y2', y1);
      grid.setAttribute('stroke-dasharray','2,3');
      gGrid.appendChild(grid);
      const lab = document.createElementNS(ns,'text');
      lab.setAttribute('x', X); lab.setAttribute('y', y0+14);
      lab.setAttribute('text-anchor','middle');
      lab.textContent = t;
      gLab.appendChild(lab);
    });
    yTicks.forEach(tv=>{
      const Y = sy(tv);
      const tick = document.createElementNS(ns,'line');
      tick.setAttribute('x1', x0-4); tick.setAttribute('y1', Y);
      tick.setAttribute('x2', x0);  tick.setAttribute('y2', Y);
      gAxis.appendChild(tick);
      const grid = document.createElementNS(ns,'line');
      grid.setAttribute('x1', x0); grid.setAttribute('y1', Y);
      grid.setAttribute('x2', x1); grid.setAttribute('y2', Y);
      grid.setAttribute('stroke-dasharray','2,3');
      gGrid.appendChild(grid);
      const lab = document.createElementNS(ns,'text');
      lab.setAttribute('x', x0-6); lab.setAttribute('y', Y+3);
      lab.setAttribute('text-anchor','end');
      lab.textContent = tv.toFixed( (Math.abs(tv)<5)?1:0 );
      gLab.appendChild(lab);
    });
    const yUnitLab = document.createElementNS(ns,'text');
    yUnitLab.setAttribute('x', 10);
    yUnitLab.setAttribute('y', 12);
    yUnitLab.setAttribute('fill','#bbb');
    yUnitLab.setAttribute('font-size','9');
    yUnitLab.setAttribute('text-anchor','start');
    yUnitLab.textContent = yUnit;
    gLab.appendChild(yUnitLab);
    const pathEl = document.createElementNS(ns,'path');
    let path=''; let penDown=false;
    data.forEach((v,i)=>{
      if (v==null || isNaN(v)){ penDown=false; return; }
      const cmd = penDown ? 'L' : 'M';
      path += `${cmd}${sx(i).toFixed(2)},${sy(v).toFixed(2)} `;
      penDown=true;
    });
    pathEl.setAttribute('d', path.trim());
    pathEl.setAttribute('fill','none');
    pathEl.setAttribute('stroke','white');
    pathEl.setAttribute('stroke-opacity','0.9');
    pathEl.setAttribute('stroke-width','1.6');
    const gPlot = document.createElementNS(ns,'g');
    gPlot.appendChild(pathEl);
    const gAll = document.createElementNS(ns,'g');
    gAll.appendChild(gGrid); gAll.appendChild(gAxis); gAll.appendChild(gLab); gAll.appendChild(gPlot);
    svgEl.appendChild(gAll);
  }

  const DAY_TEX  = 'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg';
  const BUMP_TEX = 'https://unpkg.com/three-globe/example/img/earth-topology.png';
  const BG_TEX   = 'https://unpkg.com/three-globe/example/img/night-sky.png';
  const globe = Globe({ rendererConfig: { antialias: true, alpha: true, logarithmicDepthBuffer: true } })(document.getElementById('root'))
    .globeImageUrl(DAY_TEX)
    .bumpImageUrl(BUMP_TEX)
    .backgroundImageUrl(BG_TEX)
    .showAtmosphere(true)
    .atmosphereColor('#88ccff')
    .atmosphereAltitude(0.18)
    .width(window.innerWidth)
    .height(window.innerHeight);
  globe.renderer().setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  globe.controls().autoRotate = false;
  globe.controls().autoRotateSpeed = 0.0;
  globe.controls().addEventListener('start', () => globe.controls().autoRotate = false);
  globe.controls().addEventListener('end',   () => globe.controls().autoRotate = true);

  let metric = PAYLOAD.default_metric || "anom";
  let idx = YEARS.length - 1;
  let valueMap = VALUES[metric][YEARS[idx]] || {};
  let colorScale = colorScaleFactory(metric, scheme);

  function setGradient(){
    const g = document.getElementById('gradBar');
    if (scheme==='normal'){
      g.style.background = 'linear-gradient(90deg,#2b6cff,#ffffff,#ff2b2b)';
    } else {
      g.style.background = 'linear-gradient(90deg,#3b4cc0,#f7f7f7,#b40426)';
    }
  }
  setGradient();

  function colorScaleFactory(m, sch){
    const MIN = CLIPS[m][0], MAX = CLIPS[m][1];
    return function(v){
      if (v==null || isNaN(v)) return 'rgba(120,120,120,0.10)';
      const x = Math.max(MIN, Math.min(MAX, v));
      const t = (x - MIN) / (MAX - MIN);
      if (sch==='normal'){
        const r = t<0.5 ? 2*t*255 : 255;
        const g = t<0.5 ? 2*t*255 : 2*(1-t)*255;
        const b = t<0.5 ? 255 : 2*(1-t)*255;
        return `rgba(${r|0},${g|0},${b|0},0.35)`;
      } else {
        const r = Math.round(59 + t*(180-59));
        const g = Math.round(76 + t*(4-76));
        const b = Math.round(192 + t*(38-192));
        return `rgba(${r},${Math.max(0,g)},${Math.max(0,b)},0.35)`;
      }
    }
  }

  function updateLegend(){
    document.getElementById('unit').textContent = UNITS[metric];
    const [MIN, MAX] = CLIPS[metric];
    document.getElementById('minlbl').textContent = MIN.toString();
    document.getElementById('maxlbl').textContent = MAX.toString();
    document.getElementById('btn-anom').classList.toggle('active', metric==='anom');
    document.getElementById('btn-abs').classList.toggle('active',  metric==='abs');
  }

  function applyYear(newIdx){
    idx = Math.max(0, Math.min(YEARS.length-1, newIdx));
    const key = YEARS[idx];
    valueMap = VALUES[metric][key] || {};
    document.getElementById('sel').textContent = key;
    globe
      .polygonCapColor(({properties}) => {
        const k = csvName(properties.NAME);
        const v = getValue(valueMap, k);
        return colorScale(v);
      })
      .polygonLabel(({ properties }) => String(properties.NAME || ""));
    if (selectedCountry){ openInfo(selectedCountry); }
    globe.polygonsData(globe.polygonsData());
  }

  function applyMetric(newMetric){
    metric = newMetric;
    colorScale = colorScaleFactory(metric, scheme);
    updateLegend();
    applyYear(idx);
    if (selectedCountry){ openInfo(selectedCountry); }
  }

  function openInfo(name){
    selectedCountry = name;
    const info  = document.getElementById('info');
    const title = document.getElementById('infoTitle');
    const text  = document.getElementById('infoText');
    const svg   = document.getElementById('infoSvg');
    title.textContent = name;
    const currentYear = YEARS[idx];
    const latestYear  = YEARS[YEARS.length - 1];
    const key = csvName(name);
    const currentVal =
        VALUES[metric][currentYear]?.[key]
     ?? VALUES[metric][currentYear]?.[key?.replaceAll('-', ' ')]
     ?? VALUES[metric][currentYear]?.[key?.replaceAll(' ', '-')]
     ?? null;
    const ysFull = seriesForCountry(name, metric);
    const lr = linreg(ysFull);
    const slopePerDecade = (lr.slope * 10);
    const nowStr = (metric === 'anom'
      ? (currentVal != null ? `Temperature Anomaly: ${currentVal.toFixed(2)} °C` : 'no data')
      : (currentVal != null ? `Average Temperature: ${currentVal.toFixed(1)} °C` : 'no data')
    );
    const slopeStr = `${slopePerDecade.toFixed(2)} °C/decade`;
    text.innerHTML = `
      <div><b>${currentYear}</b> snapshot: <b>${nowStr}</b></div>
      <div>Trend (linear, 1901–${latestYear}): <b>${slopeStr}</b></div>
      <div style="opacity:.8">Tip: the chart shows the full history; the snapshot follows the year slider.</div>`;
    const xTicks = [YEARS[0], '1950', '2000', latestYear];
    const valid = ysFull.filter(v => v != null && !isNaN(v));
    const ymin = Math.min(...valid), ymax = Math.max(...valid);
    const span = (ymax - ymin) || 1e-6;
    const yTicks = [ymin, ymin + span * 0.5, ymax];
    const yUnitShort = '°C';
    sparklineSVG(svg, ysFull, { xTicks, yTicks, yUnit: yUnitShort });
    info.classList.add('show');
  }

  function closeInfo(){
    document.getElementById('info').classList.remove('show');
    selectedCountry = null;
  }

  fetch('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson')
    .then(r => r.json())
    .then(geo => {
      globe
        .polygonsData(geo.features)
        .polygonAltitude(0.005)
        .polygonSideColor(() => 'rgba(0,0,0,0)')
        .polygonStrokeColor(() => 'rgba(255,255,255,0.55)')
        .onPolygonClick(({properties}) => {
          const shown = String(properties.NAME || "");
          openInfo(shown);
        });
      updateLegend();
      applyYear(idx);
    });

  document.getElementById('btn-anom').onclick = () => applyMetric('anom');
  document.getElementById('btn-abs').onclick  = () => applyMetric('abs');
  document.getElementById('range').addEventListener('input', (e) => applyYear(parseInt(e.target.value,10)));
  window.addEventListener('resize', () => { globe.width(window.innerWidth); globe.height(window.innerHeight); });

  const blog = document.getElementById('blog');
  const blogContent = document.getElementById('blogContent');
  function openBlog(){ blog.classList.add('show'); globe.controls().autoRotate = false; }
  function closeBlog(){ blog.classList.remove('show'); globe.controls().autoRotate = true; }
  document.getElementById('openBlog').onclick = openBlog;
  document.getElementById('blogClose').onclick = closeBlog;
  blogContent.innerHTML = BLOG;
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closeBlog(); closeInfo(); } });
  document.getElementById('infoClose').onclick = closeInfo;

  document.getElementById('btn-cb').onclick = () => {
    scheme = (scheme==='normal') ? 'cb' : 'normal';
    document.getElementById('btn-cb').textContent = `Colorblind: ${scheme==='cb'?'ON':'OFF'}`;
    colorScale = colorScaleFactory(metric, scheme);
    setGradient();
    applyYear(idx);
  };

  document.getElementById('btn-png').onclick = () => {
    const canvas = globe.renderer().domElement;
    const url = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = `EmissionWiz_${metric}_${YEARS[idx]}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };
</script>
</body>
</html>
"""

html(
    HTML.replace("__PAYLOAD__", PAYLOAD_JSON)
        .replace("__UNIT__", payload["units"][payload["default_metric"]])
        .replace("__MIN__", str(payload["clips"][payload["default_metric"]][0]))
        .replace("__MAX__", str(payload["clips"][payload["default_metric"]][1]))
        .replace("__MAXIDX__", str(len(payload["years"]) - 1))
        .replace("__BLOG__", BLOG_JSON),
    height=10,
    scrolling=False
)
