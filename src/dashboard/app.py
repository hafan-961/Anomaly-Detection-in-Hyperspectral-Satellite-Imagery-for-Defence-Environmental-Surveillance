"""
Stage 6 — Flask Dashboard with Spectral Signature Matching
Hyperspectral Anomaly Detection — AVIRIS-NG Mississippi Delta
Author: Muhammed Hafan — LPU B.Tech CSE 2023-2027
"""

import os
import json
import math
import numpy as np
from flask import Flask, render_template_string, jsonify, request, Response
import spectral.io.envi as envi

app = Flask(__name__)


# PATHS
BASE       = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(BASE, "..", ".."))
GEOJSON    = os.path.join(ROOT, "outputs", "anomalies.geojson")
METRICS    = os.path.join(ROOT, "outputs", "metrics.json")
SCORE_MAP  = os.path.join(ROOT, "data", "processed", "score_map_smooth.npy")
VALID_MASK = os.path.join(ROOT, "data", "processed", "valid_mask.npy")
HDR        = os.path.join(ROOT, "data", "ang20210925t151637_rfl_v2z1.hdr")
DATA       = os.path.join(ROOT, "data", "ang20210925t151637_rfl_v2z1")


# LOAD RAW HYPERSPECTRAL IMAGE (for spectral extraction)

print("Loading hyperspectral image for spectral matching...")
try:
    img = envi.open(HDR, DATA)
    wavelengths = np.array(img.bands.centers)
    IMG_LOADED = True
    print(f"  Image loaded: {img.shape}")
except Exception as e:
    img = None
    wavelengths = None
    IMG_LOADED = False
    print(f"  Warning: Could not load image: {e}")


# BUILT-IN SPECTRAL SIGNATURES
# Mississippi Delta — 6 material classes
# Signatures defined as (wavelength_range, reflectance_pattern) rules
# Using simplified spectral indices for matching


MATERIAL_SIGNATURES = {
    "Oil / Hydrocarbon Spill": {
        "emoji": "🛢️",
        "color": "#8B4513",
        "description": "Petroleum or hydrocarbon contamination on water/land surface",
        "indicators": {
            "low_nir": True,        # low reflectance 800-900nm
            "swir_absorption": True, # strong absorption 1700-2300nm
            "low_visible": True,    # dark in visible range
            "high_red": False,
        },
        "wavelength_rules": [
            (380,  700,"low",0.30),   # dark in visible
            (800,  900,"low",0.25),   # low NIR
            (1600, 1800, "low", 0.20),   # SWIR absorption
        ]
    },
    "Burn Scar / Char": {
        "emoji": "🔥",
        "color": "#FF4500",
        "description": "Post-fire burn scar from Hurricane Ida (Aug 2021 landfall)",
        "indicators": {
            "low_nir": True,
            "low_visible": True,
            "flat_spectrum": True,
        },
        "wavelength_rules": [
            (380,700,  "low",0.15),# very dark visible
            (700,900,"low",0.20),# suppressed red edge
            (1500, 2500, "low",   0.15),# dark SWIR
        ]
    },
    "Turbid / Sediment-Laden Water": {
        "emoji": "🌊",
        "color": "#4169E1",
        "description": "High sediment load — Mississippi River runoff or storm surge",
        "indicators": {
            "high_visible": True,
            "low_nir": True,
            "water_absorption": True,
        },
        "wavelength_rules": [
            (550, 700, "high",0.15),   # elevated green-red (sediment)
            (800,900,"low",0.10),   # water absorbs NIR
            (1300, 1500, "low",0.05),   # strong water absorption
        ]
    },
    "Urban / Industrial Surface": {
        "emoji": "🏭",
        "color": "#708090",
        "description": "Rooftop, road, concrete, or industrial material",
        "indicators": {
            "flat_spectrum": True,
            "medium_reflectance": True,
        },
        "wavelength_rules": [
            (380,  700,  "medium", 0.20),  # moderate visible
            (700,  1300, "medium", 0.25),  # moderate NIR
            (1500, 2500, "medium", 0.20),  # moderate SWIR
        ]
    },
    "Stressed / Sparse Vegetation": {
        "emoji": "🌿",
        "color": "#556B2F",
        "description": "Damaged or stressed marsh/wetland vegetation",
        "indicators": {
            "weak_red_edge": True,
            "low_nir": True,
        },
        "wavelength_rules": [
            (630,  690,  "low",   0.10),   # red absorption (weak)
            (700,  800,  "medium", 0.25),  # weak red edge
            (800,  900,  "medium", 0.30),  # reduced NIR plateau
        ]
    },
    "Bare Soil / Exposed Sediment": {
        "emoji": "🟫",
        "color": "#D2691E",
        "description": "Exposed delta sediment, mud flat, or disturbed ground",
        "indicators": {
            "rising_spectrum": True,
            "medium_swir": True,
        },
        "wavelength_rules": [
            (380,  700,  "medium", 0.20),  # moderate visible
            (700,  1300, "high",   0.35),  # rising NIR
            (1500, 2000, "medium", 0.25),  # moderate SWIR
        ]
    },
}


def extract_spectrum(row, col):
    """Extract 425-band spectrum for a pixel from the raw image."""
    if not IMG_LOADED or img is None:
        return None, None
    try:
        row, col = int(row), int(col)
        spectrum = img.read_pixel(row, col).astype(np.float32)
        spectrum = np.clip(spectrum, 0, 1)
        return spectrum, wavelengths
    except Exception as e:
        print(f"Spectrum extraction error: {e}")
        return None, None


def match_spectrum(spectrum, wl):
    """
    Match a spectrum against built-in material signatures.
    Returns list of (material, confidence, description) sorted by confidence.
    """
    if spectrum is None or wl is None:
        return []

    scores = {}

    for material, sig in MATERIAL_SIGNATURES.items():
        score = 0.0
        total_weight = 0.0

        for wl_min, wl_max, level, threshold in sig["wavelength_rules"]:
            # Find bands in this wavelength range
            band_mask = (wl >= wl_min) & (wl <= wl_max)
            if not band_mask.any():
                continue

            band_vals = spectrum[band_mask]
            mean_val  = float(np.mean(band_vals))
            weight    = 1.0

            if level == "low":
                match = 1.0 - min(mean_val / (threshold * 2 + 1e-9), 1.0)
                if mean_val < threshold:
                    match = min(match * 1.5, 1.0)
            elif level == "high":
                match = min(mean_val / (threshold + 1e-9), 1.0)
                if mean_val > threshold:
                    match = min(match * 1.3, 1.0)
            elif level == "medium":
                dist  = abs(mean_val - threshold)
                match = max(0.0, 1.0 - dist / (threshold + 1e-9))

            score        += match * weight
            total_weight += weight

        if total_weight > 0:
            scores[material] = score / total_weight

    # Normalize to 0-100% confidence
    max_score = max(scores.values()) if scores else 1.0
    results = []
    for material, score in sorted(scores.items(), key=lambda x: -x[1]):
        confidence = min(int((score / (max_score + 1e-9)) * 100), 99)
        results.append({
            "material":    material,
            "confidence":  confidence,
            "emoji":       MATERIAL_SIGNATURES[material]["emoji"],
            "color":       MATERIAL_SIGNATURES[material]["color"],
            "description": MATERIAL_SIGNATURES[material]["description"],
        })

    return results[:3]  # top 3



# UTM → LAT/LON CONVERSION

def utm_to_latlon(easting, northing, zone=15, northern=True):
    a  = 6378137.0
    f  = 1 / 298.257223563
    b  = a * (1 - f)
    e2 = 1 - (b ** 2) / (a ** 2)
    k0 = 0.9996
    E0 = 500000.0
    N0 = 0.0 if northern else 10000000.0
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    N  = northing - N0
    E  = easting  - E0
    M  = N / k0
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    e1 = (1 - math.sqrt(1-e2)) / (1 + math.sqrt(1-e2))
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



# HTML TEMPLATE

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

    #main { display:flex; flex:1; overflow:hidden; }

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

    /* Spectral match panel */
    #spectral-panel { margin-top:10px; }
    #spectral-panel h4 {
      font-size:11px; font-weight:700; color:#5588bb;
      text-transform:uppercase; letter-spacing:.8px; margin-bottom:8px;
    }
    .match-card {
      background:#1a2235; border-radius:8px; padding:10px 12px;
      border:1px solid #243050; margin-bottom:6px;
    }
    .match-card.rank-1 { border-color:#f39c12; }
    .match-card.rank-2 { border-color:#3a5a8c; }
    .match-card.rank-3 { border-color:#2a3a4c; }
    .match-header {
      display:flex; align-items:center; justify-content:space-between;
      margin-bottom:5px;
    }
    .match-name { font-size:12px; font-weight:700; color:#e0e0e0; }
    .match-confidence {
      font-size:13px; font-weight:700; padding:2px 7px;
      border-radius:4px; background:#243050;
    }
    .match-desc { font-size:10px; color:#6688aa; margin-bottom:6px; }
    .conf-bar-bg { background:#243050; border-radius:4px; height:5px; }
    .conf-bar-fill { border-radius:4px; height:5px; transition:width .5s; }
    .match-loading {
      color:#445566; font-style:italic; text-align:center;
      padding:12px 0; font-size:11px;
    }

    /* Spectral chart */
    #spectral-chart-section { margin-top:10px; }
    #spectral-chart-section h4 {
      font-size:11px; font-weight:700; color:#5588bb;
      text-transform:uppercase; letter-spacing:.8px; margin-bottom:8px;
    }
    #spectral-canvas {
      width:100%; height:120px; background:#1a2235;
      border-radius:8px; border:1px solid #243050;
    }

    #statusbar {
      padding:4px 16px; font-size:10px; color:#445566;
      border-top:1px solid #1e2d45; margin-top:auto;
    }
    #statusbar span { color:#5588bb; }

    #map { flex:1; }

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

      <!-- Spectral Signature Match -->
      <div id="spectral-panel" style="display:none;">
        <h4>🔬 Spectral Signature Match</h4>
        <div id="match-results">
          <div class="match-loading">Analyzing spectrum...</div>
        </div>
      </div>

      <!-- Spectral Chart -->
      <div id="spectral-chart-section" style="display:none;">
        <h4>📈 Spectral Profile</h4>
        <canvas id="spectral-canvas"></canvas>
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

function scoreColor(score) {
  const norm = Math.min(score / 0.12, 1.0);
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
      radius: 5, fillColor: scoreColor(pt.score),
      color: '#fff', weight: 0.5, fillOpacity: 0.9, opacity: 1.0
    });
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
    circle.on('click', () => onMarkerClick(pt));
    circle.addTo(markerLayer);
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

// ── MARKER CLICK → SPECTRAL MATCH ────────
function onMarkerClick(pt) {
  showPixelInfo(pt);
  document.getElementById('spectral-panel').style.display = 'block';
  document.getElementById('match-results').innerHTML =
    '<div class="match-loading">🔬 Analyzing spectral signature...</div>';

  fetch(`/api/spectral?row=${pt.row}&col=${pt.col}`)
    .then(r => r.json())
    .then(data => {
      renderMatches(data.matches);
      if (data.spectrum && data.wavelengths) {
        renderSpectralChart(data.spectrum, data.wavelengths);
      }
    })
    .catch(() => {
      document.getElementById('match-results').innerHTML =
        '<div class="match-loading">⚠️ Spectrum unavailable</div>';
    });
}

// ── RENDER SPECTRAL MATCHES ───────────────
function renderMatches(matches) {
  if (!matches || matches.length === 0) {
    document.getElementById('match-results').innerHTML =
      '<div class="match-loading">No matches found</div>';
    return;
  }
  const rankClass = ['rank-1', 'rank-2', 'rank-3'];
  const rankLabel = ['#1 Best Match', '#2', '#3'];
  let html = '';
  matches.forEach((m, i) => {
    html += `
      <div class="match-card ${rankClass[i]}">
        <div class="match-header">
          <span class="match-name">${m.emoji} ${m.material}</span>
          <span class="match-confidence" style="color:${m.color}">${m.confidence}%</span>
        </div>
        <div class="match-desc">${m.description}</div>
        <div class="conf-bar-bg">
          <div class="conf-bar-fill" style="width:${m.confidence}%; background:${m.color}"></div>
        </div>
      </div>
    `;
  });
  document.getElementById('match-results').innerHTML = html;
}

// ── SPECTRAL CHART ────────────────────────
function renderSpectralChart(spectrum, wavelengths) {
  const section = document.getElementById('spectral-chart-section');
  section.style.display = 'block';
  const canvas = document.getElementById('spectral-canvas');
  const ctx    = canvas.getContext('2d');
  canvas.width  = canvas.offsetWidth * window.devicePixelRatio;
  canvas.height = canvas.offsetHeight * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  const W = canvas.offsetWidth;
  const H = canvas.offsetHeight;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#1a2235';
  ctx.fillRect(0, 0, W, H);

  const pad = { top:10, right:10, bottom:20, left:35 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top  - pad.bottom;

  const minWL = Math.min(...wavelengths);
  const maxWL = Math.max(...wavelengths);
  const maxRef = Math.max(...spectrum) || 1;

  // Grid lines
  ctx.strokeStyle = '#243050';
  ctx.lineWidth   = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (chartH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + chartW, y);
    ctx.stroke();
  }

  // Y axis labels
  ctx.fillStyle = '#445566';
  ctx.font      = '8px Segoe UI';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = ((4 - i) / 4 * maxRef).toFixed(2);
    const y   = pad.top + (chartH / 4) * i;
    ctx.fillText(val, pad.left - 3, y + 3);
  }

  // X axis labels
  ctx.textAlign = 'center';
  const wlLabels = [400, 800, 1200, 1600, 2000, 2400];
  wlLabels.forEach(wl => {
    if (wl < minWL || wl > maxWL) return;
    const x = pad.left + ((wl - minWL) / (maxWL - minWL)) * chartW;
    ctx.fillText(wl, x, H - 4);
  });

  // Spectrum line
  const gradient = ctx.createLinearGradient(pad.left, 0, pad.left + chartW, 0);
  gradient.addColorStop(0.0,  '#8B00FF');
  gradient.addColorStop(0.2,  '#0000FF');
  gradient.addColorStop(0.35, '#00FF00');
  gradient.addColorStop(0.5,  '#FFFF00');
  gradient.addColorStop(0.65, '#FF0000');
  gradient.addColorStop(1.0,  '#8B0000');

  ctx.beginPath();
  ctx.strokeStyle = gradient;
  ctx.lineWidth   = 1.5;
  spectrum.forEach((val, i) => {
    const x = pad.left + ((wavelengths[i] - minWL) / (maxWL - minWL)) * chartW;
    const y = pad.top  + chartH - (val / maxRef) * chartH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
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
  document.getElementById('m-auc').textContent       = m.ROC_AUC_VAE?.toFixed(3)  ?? '—';
  document.getElementById('m-prauc').textContent     = m.PR_AUC_VAE?.toFixed(3)   ?? '—';
  document.getElementById('m-f1').textContent        = m.F1?.toFixed(3)            ?? '—';
  document.getElementById('m-precision').textContent = m.Precision?.toFixed(3)     ?? '—';
  document.getElementById('m-recall').textContent    = m.Recall?.toFixed(3)        ?? '—';
  document.getElementById('m-total').textContent     =
    (m.anomaly_pixels_filtered ?? m.TP ?? '—').toLocaleString();
  document.getElementById('m-noise').textContent     =
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
const slider        = document.getElementById('threshold-slider');
const threshDisplay = document.getElementById('threshold-val');
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
renderMarkers(0);
setTimeout(resetView, 300);
</script>
</body>
</html>
"""

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
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

@app.route("/api/spectral")
def api_spectral():
    """Extract spectrum and return material matches."""
    row = int(request.args.get("row", 0))
    col = int(request.args.get("col", 0))

    spectrum, wl = extract_spectrum(row, col)
    matches      = match_spectrum(spectrum, wl)

    return jsonify({
        "row":         row,
        "col":         col,
        "matches":     matches,
        "spectrum":    spectrum.tolist() if spectrum is not None else None,
        "wavelengths": wl.tolist()       if wl is not None       else None,
    })

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

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("STAGE 6 — FLASK DASHBOARD + SPECTRAL MATCHING")
    print("=" * 50)
    print(f"\n  Loading data from: {ROOT}")
    try:
        metrics, features = load_data()
        print(f"  Metrics loaded:    ✅")
        print(f"  Anomaly points:    {len(features):,}")
        print(f"  Spectral matching: {'✅ Ready' if IMG_LOADED else '⚠️ Image not found'}")
        if features:
            print(f"  Lat range:  {min(f['lat'] for f in features):.4f} → {max(f['lat'] for f in features):.4f}")
            print(f"  Lon range:  {min(f['lon'] for f in features):.4f} → {max(f['lon'] for f in features):.4f}")
            print(f"  Score range:{min(f['score'] for f in features):.4f} → {max(f['score'] for f in features):.4f}")
        else:
            print("  ⚠️  No features passed sanity check")
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
    print(f"\n  → http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)