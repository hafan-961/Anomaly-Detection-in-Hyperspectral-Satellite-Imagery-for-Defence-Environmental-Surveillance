"""
Stage 6 — Flask Dashboard
Hyperspectral Anomaly Detection — AVIRIS-NG Mississippi Delta
Author: Muhammed Hafan — LPU B.Tech CSE 2023-2027
"""

import os
import json
import math
import numpy as np
from flask import Flask, render_template_string, jsonify, request, Response

app = Flask(__name__)


# PATHS

BASE= os.path.dirname(os.path.abspath(__file__))
ROOT= os.path.abspath(os.path.join(BASE, "..", ".."))
GEOJSON= os.path.join(ROOT, "outputs", "anomalies.geojson")
METRICS = os.path.join(ROOT, "outputs", "metrics.json")
SCORE_MAP = os.path.join(ROOT, "data", "processed", "score_map_smooth.npy")
VALID_MASK = os.path.join(ROOT, "data", "processed", "valid_mask.npy")


# UTM → LAT/LON CONVERSION
def utm_to_latlon(easting, northing, zone=15, northern=True):
    a  = 6378137.0
    f  = 1 / 298.257223563
    b  = a * (1 - f)
    e2 = 1 - (b ** 2) / (a ** 2)

    k0   = 0.9996
    E0   = 500000.0
    N0   = 0.0 if northern else 10000000.0
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)

    N  = northing - N0
    E  = easting  - E0
    M  = N / k0
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))

    e1   = (1 - math.sqrt(1-e2)) / (1 + math.sqrt(1-e2))
    phi1 = (mu
            + (3*e1/2 - 27*e1**3/32) * math.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu)
            + (151*e1**3/96) * math.sin(6*mu))

    N1 = a / math.sqrt(1 - e2 * math.sin(phi1)**2)
    T1 = math.tan(phi1)**2
    C1 = e2 / (1-e2) * math.cos(phi1)**2
    R1 = a*(1-e2) / (1 - e2*math.sin(phi1)**2)**1.5
    D  = E / (N1 * k0)

    lat = phi1 - (N1*math.tan(phi1)/R1) * (
        D**2/2
        - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e2/(1-e2)) * D**4/24
        + (61 + 90*T1 + 298*C1 + 45*T1**2 - 3*C1**2 - 252*e2/(1-e2)) * D**6/720
    )
    lon = lon0 + (
        D
        - (1 + 2*T1 + C1) * D**3/6
        + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*e2/(1-e2) + 24*T1**2) * D**5/120
    ) / math.cos(phi1)

    return math.degrees(lat), math.degrees(lon)



# DATA LOADING

def load_data():
    with open(METRICS) as f:
        metrics = json.load(f)

    with open(GEOJSON) as f:
        raw = json.load(f)

    features = []
    for feat in raw["features"]:
        props = feat["properties"]
        col   = props.get("col", 0)
        row   = props.get("row", 0)

        # Convert pixel row/col → UTM → lat/lon
        # HDR: UTM Zone 15N, UL=(773286.108883, 3238616.43099), res=4.8m, rot=38°
        angle   = math.radians(38.0)
        px_size = 4.8
        ul_e    = 773286.108883
        ul_n    = 3238616.43099

        easting  = ul_e + (col * px_size * math.cos(angle)) - (row * px_size * math.sin(angle))
        northing = ul_n - (col * px_size * math.sin(angle)) - (row * px_size * math.cos(angle))

        try:
            lat, lon = utm_to_latlon(easting, northing, zone=15, northern=True)
        except Exception:
            continue

        # Sanity check — Mississippi Delta region
        if not (-95 < lon < -85 and 27 < lat < 35):
            continue

        features.append({
            "lat":   round(lat, 6),
            "lon":   round(lon, 6),
            "score": round(float(props.get("score", 0)), 4),
            "row":   int(row),
            "col":   int(col),
        })

    return metrics, features





TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Hyperspectral Anomaly Detection — AVIRIS-NG</title>

  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>

  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { font-family:'Segoe UI',sans-serif; background:#0f1117; color:#e0e0e0;
           display:flex; flex-direction:column; height:100vh; overflow:hidden; }

    /* Navbar */
    #navbar {
      background:linear-gradient(90deg,#1a1f2e,#16213e);
      padding:10px 20px; display:flex; align-items:center;
      justify-content:space-between; border-bottom:1px solid #2a3a5c; flex-shrink:0;
    }
    #navbar .brand { display:flex; align-items:center; gap:10px; }
    #navbar .brand-icon {
      width:34px; height:34px; border-radius:8px;
      background:linear-gradient(135deg,#e74c3c,#f39c12);
      display:flex; align-items:center; justify-content:center; font-size:18px;
    }
    #navbar .brand-text h1 { font-size:15px; font-weight:700; color:#fff; }
    #navbar .brand-text p  { font-size:11px; color:#8899aa; }
    #navbar .nav-btns { display:flex; gap:8px; }
    .nav-btn {
      padding:6px 14px; border-radius:6px; border:none; cursor:pointer;
      font-size:12px; font-weight:600; transition:all .2s;
    }
    .nav-btn.primary   { background:#e74c3c; color:#fff; }
    .nav-btn.primary:hover { background:#c0392b; }
    .nav-btn.secondary { background:#2a3a5c; color:#cdd; border:1px solid #3a5a8c; }
    .nav-btn.secondary:hover { background:#3a4a6c; }

    /* Main */
    #main { display:flex; flex:1; overflow:hidden; }

    /* Sidebar */
    #sidebar {
      width:320px; min-width:320px; background:#13192a;
      border-right:1px solid #1e2d45; display:flex;
      flex-direction:column; overflow-y:auto;
    }
    .sidebar-section { padding:14px 16px; border-bottom:1px solid #1e2d45; }
    .sidebar-section h3 {
      font-size:11px; font-weight:700; color:#5588bb;
      text-transform:uppercase; letter-spacing:.8px; margin-bottom:10px;
    }

    /* Metric cards */
    .metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .metric-card {
      background:#1a2235; border-radius:8px; padding:10px 12px;
      border:1px solid #243050;
    }
    .metric-card .label { font-size:10px; color:#6688aa; margin-bottom:3px; }
    .metric-card .value { font-size:20px; font-weight:700; }
    .metric-card .value.green  { color:#2ecc71; }
    .metric-card .value.orange { color:#f39c12; }
    .metric-card .value.red    { color:#e74c3c; }
    .metric-card .value.blue   { color:#3498db; }
    .metric-card .sub { font-size:10px; color:#556677; margin-top:2px; }

    /* Slider */
    #threshold-val {
      font-size:22px; font-weight:700; color:#f39c12;
      text-align:center; margin:6px 0;
    }
    #threshold-slider { width:100%; accent-color:#f39c12; cursor:pointer; }
    .slider-labels {
      display:flex; justify-content:space-between;
      font-size:10px; color:#556677; margin-top:3px;
    }
    #detection-count { text-align:center; font-size:12px; color:#8899bb; margin-top:6px; }

    /* Toggles */
    .toggle-row {
      display:flex; align-items:center; justify-content:space-between; padding:6px 0;
    }
    .toggle-row label { font-size:12px; color:#aabbcc; }
    .toggle { position:relative; width:38px; height:20px; cursor:pointer; }
    .toggle input { opacity:0; width:0; height:0; }
    .toggle-slider {
      position:absolute; inset:0; background:#2a3a5c;
      border-radius:20px; transition:.3s;
    }
    .toggle-slider:before {
      content:""; position:absolute; width:14px; height:14px;
      left:3px; top:3px; background:#fff; border-radius:50%; transition:.3s;
    }
    .toggle input:checked + .toggle-slider { background:#e74c3c; }
    .toggle input:checked + .toggle-slider:before { transform:translateX(18px); }

    /* Pixel info */
    #pixel-info {
      background:#1a2235; border-radius:8px; padding:12px;
      border:1px solid #243050; font-size:12px;
    }
    #pixel-info .pi-row {
      display:flex; justify-content:space-between; padding:3px 0;
      border-bottom:1px solid #1e2d45;
    }
    #pixel-info .pi-row:last-child { border:none; }
    #pixel-info .pi-key   { color:#6688aa; }
    #pixel-info .pi-value { color:#e0e0e0; font-weight:600; }
    #pixel-info .pi-empty {
      color:#445566; font-style:italic; text-align:center; padding:8px 0;
    }

    /* Status */
    #statusbar {
      padding:4px 16px; font-size:10px; color:#445566;
      border-top:1px solid #1e2d45; margin-top:auto;
    }
    #statusbar span { color:#5588bb; }

    /* Map */
    #map { flex:1; }

    /* Popup */
    .leaflet-popup-content-wrapper {
      background:#1a2235 !important; color:#e0e0e0 !important;
      border:1px solid #3a5a8c !important; border-radius:8px !important;
    }
    .leaflet-popup-tip { background:#1a2235 !important; }
    .popup-content h4 { color:#f39c12; margin-bottom:6px; font-size:13px; }
    .popup-row {
      display:flex; justify-content:space-between; gap:20px;
      font-size:12px; padding:2px 0;
    }
    .popup-key { color:#6688aa; }
    .popup-val { color:#e0e0e0; font-weight:600; }
    .score-bar-bg { background:#243050; border-radius:4px; height:6px; margin-top:6px; }
    .score-bar-fill {
      background:linear-gradient(90deg,#f39c12,#e74c3c);
      border-radius:4px; height:6px;
    }
  </style>
</head>
<body>

<div id="navbar">
  <div class="brand">
    <div class="brand-icon">🛰️</div>
    <div class="brand-text">
      <h1>Hyperspectral Anomaly Detection</h1>
      <p>NASA AVIRIS-NG · Mississippi River Delta · 2021</p>
    </div>
  </div>
  <div class="nav-btns">
    <button class="nav-btn secondary" onclick="exportGeoJSON()">⬇ GeoJSON</button>
    <button class="nav-btn secondary" onclick="exportKML()">⬇ KML</button>
    <button class="nav-btn primary"   onclick="resetView()">⌖ Reset View</button>
  </div>
</div>

<div id="main">
  <div id="sidebar">

    <div class="sidebar-section">
      <h3>Model Performance</h3>
      <div class="metric-grid">
        <div class="metric-card">
          <div class="label">ROC-AUC (VAE)</div>
          <div class="value green" id="m-auc">—</div>
          <div class="sub">Primary metric</div>
        </div>
        <div class="metric-card">
          <div class="label">PR-AUC (VAE)</div>
          <div class="value blue" id="m-prauc">—</div>
          <div class="sub">vs random baseline</div>
        </div>
        <div class="metric-card">
          <div class="label">F1 Score</div>
          <div class="value orange" id="m-f1">—</div>
          <div class="sub">Geo-mean fusion</div>
        </div>
        <div class="metric-card">
          <div class="label">PFAR</div>
          <div class="value" id="m-pfar">—</div>
          <div class="sub">False alarm rate</div>
        </div>
        <div class="metric-card">
          <div class="label">Precision</div>
          <div class="value" id="m-precision">—</div>
          <div class="sub"></div>
        </div>
        <div class="metric-card">
          <div class="label">Recall</div>
          <div class="value" id="m-recall">—</div>
          <div class="sub">Detection rate</div>
        </div>
      </div>
    </div>

    <div class="sidebar-section">
      <h3>Detection Summary</h3>
      <div class="metric-grid">
        <div class="metric-card">
          <div class="label">Anomaly Pixels</div>
          <div class="value orange" id="m-total">—</div>
          <div class="sub">After filtering</div>
        </div>
        <div class="metric-card">
          <div class="label">Noise Removed</div>
          <div class="value red" id="m-noise">—</div>
          <div class="sub">Cluster filter</div>
        </div>
      </div>
    </div>

    <div class="sidebar-section">
      <h3>Detection Threshold</h3>
      <div id="threshold-val">0.000</div>
      <input type="range" id="threshold-slider"
             min="0" max="0.15" step="0.001" value="0"/>
      <div class="slider-labels"><span>Sensitive</span><span>Strict</span></div>
      <div id="detection-count">— detections visible</div>
    </div>

    <div class="sidebar-section">
      <h3>Map Layers</h3>
      <div class="toggle-row">
        <label>Anomaly Markers</label>
        <label class="toggle">
          <input type="checkbox" id="tog-markers" checked onchange="toggleLayer('markers')"/>
          <span class="toggle-slider"></span>
        </label>
      </div>
      <div class="toggle-row">
        <label>Heat Map</label>
        <label class="toggle">
          <input type="checkbox" id="tog-heat" checked onchange="toggleLayer('heat')"/>
          <span class="toggle-slider"></span>
        </label>
      </div>
      <div class="toggle-row">
        <label>Satellite Basemap</label>
        <label class="toggle">
          <input type="checkbox" id="tog-sat" checked onchange="toggleLayer('satellite')"/>
          <span class="toggle-slider"></span>
        </label>
      </div>
    </div>

    <div class="sidebar-section">
      <h3>Selected Pixel Info</h3>
      <div id="pixel-info">
        <div class="pi-empty">Click a marker to inspect</div>
      </div>
    </div>

    <div id="statusbar">
      Dataset: <span>AVIRIS-NG L2 2021</span> ·
      Scene: <span>Mississippi Delta, LA</span> ·
      Resolution: <span>4.8m</span>
    </div>

  </div>

  <div id="map"></div>
</div>

<script>
const ANOMALY_DATA = {{ anomaly_data | tojson }};
const METRICS_DATA = {{ metrics | tojson }};

// ── MAP ──────────────────────────────────
const map = L.map('map', { zoomControl:true }).setView([29.21, -90.17], 14);

const satelliteTile = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  { attribution:'ESRI World Imagery', maxZoom:19 }
).addTo(map);

let markerLayer = L.layerGroup().addTo(map);
let heatLayer   = null;

// ── SCORE COLOR ──────────────────────────
function scoreColor(score) {
  const maxScore = 0.12;
  const norm = Math.min(score / maxScore, 1.0);
  if (norm >= 0.85) return '#e74c3c';
  if (norm >= 0.65) return '#e67e22';
  if (norm >= 0.45) return '#f1c40f';
  if (norm >= 0.25) return '#2ecc71';
  return '#3498db';
}

// ── RENDER MARKERS ───────────────────────
function renderMarkers(threshold) {
  markerLayer.clearLayers();
  const heatPts = [];
  let count = 0;

  for (const pt of ANOMALY_DATA) {
    if (pt.score < threshold) continue;
    count++;

    const circle = L.circleMarker([pt.lat, pt.lon], {
      radius: 5,
      fillColor: scoreColor(pt.score),
      color: '#fff',
      weight: 0.5,
      fillOpacity: 0.9,
      opacity: 1.0
    });

    // Score bar fill: normalize to actual data range 0.03–0.10
    const barPct = Math.min(((pt.score - 0.02) / 0.08) * 100, 100).toFixed(1);

    circle.bindPopup(`
      <div class="popup-content">
        <h4>🔴 Anomaly Detected</h4>
        <div class="popup-row">
          <span class="popup-key">Score</span>
          <span class="popup-val">${pt.score.toFixed(4)}</span>
        </div>
        <div class="popup-row">
          <span class="popup-key">Pixel</span>
          <span class="popup-val">row=${pt.row}, col=${pt.col}</span>
        </div>
        <div class="popup-row">
          <span class="popup-key">Lat / Lon</span>
          <span class="popup-val">${pt.lat.toFixed(5)}, ${pt.lon.toFixed(5)}</span>
        </div>
        <div class="score-bar-bg">
          <div class="score-bar-fill" style="width:${barPct}%"></div>
        </div>
      </div>
    `);

    circle.on('click', () => showPixelInfo(pt));
    circle.addTo(markerLayer);

    // Normalize heat intensity to actual score range
    const intensity = Math.min((pt.score - 0.02) / 0.08, 1.0);
    heatPts.push([pt.lat, pt.lon, intensity]);
  }

  if (heatLayer) map.removeLayer(heatLayer);
  if (heatPts.length > 0) {
    heatLayer = L.heatLayer(heatPts, {
      radius: 20, blur: 15, maxZoom: 17, max: 1.0,
      gradient: {0.2:'blue', 0.4:'lime', 0.6:'yellow', 0.8:'orange', 1.0:'red'}
    });
    if (document.getElementById('tog-heat').checked) heatLayer.addTo(map);
  }

  document.getElementById('detection-count').textContent =
    `${count.toLocaleString()} detections visible`;
}

// ── PIXEL INFO ───────────────────────────
function showPixelInfo(pt) {
  document.getElementById('pixel-info').innerHTML = `
    <div class="pi-row">
      <span class="pi-key">Anomaly Score</span>
      <span class="pi-value" style="color:#f39c12">${pt.score.toFixed(4)}</span>
    </div>
    <div class="pi-row">
      <span class="pi-key">Row / Col</span>
      <span class="pi-value">${pt.row} / ${pt.col}</span>
    </div>
    <div class="pi-row">
      <span class="pi-key">Latitude</span>
      <span class="pi-value">${pt.lat.toFixed(5)}°</span>
    </div>
    <div class="pi-row">
      <span class="pi-key">Longitude</span>
      <span class="pi-value">${pt.lon.toFixed(5)}°</span>
    </div>
    <div class="pi-row">
      <span class="pi-key">Sensor</span>
      <span class="pi-value">AVIRIS-NG 2021</span>
    </div>
    <div class="pi-row">
      <span class="pi-key">Resolution</span>
      <span class="pi-value">4.8m / pixel</span>
    </div>
  `;
}

// ── METRICS ──────────────────────────────
function loadMetrics() {
  const m = METRICS_DATA;
  document.getElementById('m-auc').textContent      = m.ROC_AUC_VAE?.toFixed(3)  ?? '—';
  document.getElementById('m-prauc').textContent    = m.PR_AUC_VAE?.toFixed(3)   ?? '—';
  document.getElementById('m-f1').textContent       = m.F1?.toFixed(3)            ?? '—';
  document.getElementById('m-precision').textContent= m.Precision?.toFixed(3)     ?? '—';
  document.getElementById('m-recall').textContent   = m.Recall?.toFixed(3)        ?? '—';
  document.getElementById('m-total').textContent    =
    (m.anomaly_pixels_filtered ?? m.TP ?? '—').toLocaleString();
  document.getElementById('m-noise').textContent    =
    (m.noise_pixels_removed ?? '—').toLocaleString();

  const pfar   = m.PFAR ?? 0;
  const pfarEl = document.getElementById('m-pfar');
  pfarEl.textContent = pfar.toFixed(3);
  pfarEl.className   = 'value ' + (pfar <= 0.05 ? 'green' : pfar <= 0.20 ? 'orange' : 'red');
}

// ── LAYER TOGGLES ────────────────────────
function toggleLayer(which) {
  if (which === 'markers') {
    document.getElementById('tog-markers').checked
      ? map.addLayer(markerLayer) : map.removeLayer(markerLayer);
  }
  if (which === 'heat' && heatLayer) {
    document.getElementById('tog-heat').checked
      ? map.addLayer(heatLayer) : map.removeLayer(heatLayer);
  }
  if (which === 'satellite') {
    document.getElementById('tog-sat').checked
      ? map.addLayer(satelliteTile) : map.removeLayer(satelliteTile);
  }
}

// ── THRESHOLD SLIDER ─────────────────────
const slider       = document.getElementById('threshold-slider');
const threshDisplay= document.getElementById('threshold-val');

slider.addEventListener('input', () => {
  const val = parseFloat(slider.value);
  threshDisplay.textContent = val.toFixed(3);
  renderMarkers(val);
});

// ── EXPORTS ──────────────────────────────
function exportGeoJSON() { window.location.href = '/export/geojson'; }
function exportKML()     { window.location.href = '/export/kml'; }

// ── RESET VIEW ───────────────────────────
function resetView() {
  if (ANOMALY_DATA.length > 0) {
    const lats   = ANOMALY_DATA.map(d => d.lat);
    const lons   = ANOMALY_DATA.map(d => d.lon);
    const bounds = L.latLngBounds(
      [Math.min(...lats), Math.min(...lons)],
      [Math.max(...lats), Math.max(...lons)]
    );
    map.fitBounds(bounds, { padding:[40,40] });
  } else {
    map.setView([29.21, -90.17], 14);
  }
}

// ── INIT ─────────────────────────────────
loadMetrics();
renderMarkers(0);       // ← show all markers immediately at threshold 0
setTimeout(resetView, 300);
</script>
</body>
</html>
"""


#routes

@app.route("/")
def index():
    metrics, features = load_data()
    return render_template_string(TEMPLATE, anomaly_data=features, metrics=metrics)

@app.route("/api/metrics")
def api_metrics():
    with open(METRICS) as f:
        return jsonify(json.load(f))

@app.route("/api/anomalies")
def api_anomalies():
    threshold = float(request.args.get("threshold", 0.0))
    _, features = load_data()
    filtered = [f for f in features if f["score"] >= threshold]
    return jsonify({"count": len(filtered), "features": filtered})

@app.route("/export/geojson")
def export_geojson():
    with open(GEOJSON) as f:
        data = f.read()
    return Response(data, mimetype="application/geo+json",
                    headers={"Content-Disposition": "attachment; filename=anomalies.geojson"})

@app.route("/export/kml")
def export_kml():
    _, features = load_data()
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
        '<name>AVIRIS-NG Anomalies — Mississippi Delta</name>',
        '<Style id="anomaly"><IconStyle><color>ff0000ff</color>'
        '<scale>0.6</scale></IconStyle></Style>',
    ]
    for f in features:
        lines.append(
            f'<Placemark><styleUrl>#anomaly</styleUrl>'
            f'<description>Score:{f["score"]:.4f} Row:{f["row"]} Col:{f["col"]}</description>'
            f'<Point><coordinates>{f["lon"]},{f["lat"]},0</coordinates></Point>'
            f'</Placemark>'
        )
    lines += ['</Document></kml>']
    return Response("\n".join(lines),
                    mimetype="application/vnd.google-earth.kml+xml",
                    headers={"Content-Disposition": "attachment; filename=anomalies.kml"})


# MAIN
if __name__ == "__main__":
    print("=" * 50)
    print("STAGE 6 — FLASK DASHBOARD")
    print("=" * 50)
    print(f"\n  Loading data from: {ROOT}")
    try:
        metrics, features = load_data()
        print(f"  Metrics loaded:  ✅")
        print(f"  Anomaly points:  {len(features):,}")
        if features:
            print(f"  Lat range:       {min(f['lat'] for f in features):.4f} → {max(f['lat'] for f in features):.4f}")
            print(f"  Lon range:       {min(f['lon'] for f in features):.4f} → {max(f['lon'] for f in features):.4f}")
            print(f"  Score range:     {min(f['score'] for f in features):.4f} → {max(f['score'] for f in features):.4f}")
        else:
            print("  ⚠️  No features passed sanity check")
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
    print(f"\n  → http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)