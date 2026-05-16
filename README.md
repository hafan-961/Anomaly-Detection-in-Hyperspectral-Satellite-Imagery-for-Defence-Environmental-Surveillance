# 🛰️HSAD-Intelligence
---
## Anomaly-Detection-in-Hyperspectral-Satellite-Imagery-for-Defence-Environmental-Surveillance
---

<img width="1439" height="816" alt="Screenshot 2026-05-16 at 9 36 43 PM" src="https://github.com/user-attachments/assets/b3e9fd36-5ed7-40be-bcb6-01554d37db58" />

---
## Description

HSAD-Intelligence is an unsupervised deep-learning hyperspectral anomaly detection system trained on the NASA AVIRIS-NG dataset. It is designed to identify suspicious activities by detecting anomalies for defense surveillance applications, including camouflage(materials/colours that soldiers use to make themselves and their equipment difficult to see) detection and material identification, as well as environmental monitoring  such as pollution plume detection, oil sprill ,illegal discharge identification, and vegetation stress

The model is trained on Autoencoder (AE) and a Variational Autoencoder (VAE), enabling high-precision anomaly localisation without any labelled training data

> **Key Insight:** Anomalous pixels fail to reconstruct accurately from a model trained on "normal" background spectra. The ensemble amplifies this signal while suppressing false positives from either model alone.

---
## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **Deep Ensemble** | AE + subclassed VAE with geometric mean fusion for robust anomaly scoring |
| 📉 **PCA Preprocessing** | Dimensionality reduction to 30 components, preserving >99% spectral variance |
| 🎯 **Precision-Constrained Thresholding** | Score threshold tuned to maximise precision at a configurable recall floor |
| 🌍 **Geospatial Dashboard** | Flask + Folium interactive map with anomaly overlays and coordinate display |
| 🔭 **Spectral Signature Matching** | Click any flagged pixel to view and compare its spectral curve against reference signatures |

---

## 🏗️ Architecture

```text
Raw AVIRIS-NG L2 ENVI Data (.img + .hdr)
                    │
                    ▼
┌───────────────────────────────────────────────┐
│           Preprocessing Pipeline              │
├───────────────────────────────────────────────┤
│ • Band loading using GDAL                     │
│ • NaN / Inf masking                           │
│ • L2 normalization                            │
│ • PCA dimensionality reduction                │
│ • Reduced to 30 components                    │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
                 (N_pixels × 30)
                        │
            ┌───────────┴───────────┐
            ▼                       ▼

┌────────────────────┐   ┌────────────────────┐
│   Autoencoder      │   │ Variational AE     │
│       (AE)         │   │      (VAE)         │
├────────────────────┤   ├────────────────────┤
│ • Encoder          │   │ • Encoder          │
│ • Latent Space     │   │ • μ and σ          │
│ • Decoder          │   │ • Reparameterize   │
│                    │   │ • Decoder          │
└─────────┬──────────┘   └─────────┬──────────┘
          │                        │
          ▼                        ▼

 Reconstruction Error     Reconstruction Error
          │                        │
          └──────────┬─────────────┘
                     ▼

┌───────────────────────────────────────────────┐
│        Geometric Mean Fusion Engine           │
├───────────────────────────────────────────────┤
│ anomaly_score = √(err_AE × err_VAE)           │
└───────────────────────┬───────────────────────┘
                        │
                        ▼

┌───────────────────────────────────────────────┐
│     Precision-Constrained Thresholding        │
│                     θ                         │
└───────────────────────┬───────────────────────┘
                        │
               ┌────────┴────────┐
               ▼                 ▼

      ┌────────────────┐ ┌────────────────┐
      │ Anomalous      │ │ Normal         │
      │ Pixels         │ │ Pixels         │
      └───────┬────────┘ └────────────────┘
              │
              ▼

┌───────────────────────────────────────────────┐
│          Geospatial Mapping                   │
│            using Folium Maps                  │
└───────────────────────┬───────────────────────┘
                        │
                        ▼

┌───────────────────────────────────────────────┐
│      Flask-Based Interactive Dashboard        │
├───────────────────────────────────────────────┤
│ • Real-time visualization                     │
│ • Spectral signature matching                 │
│ • Threat highlighting                         │
│ • Interactive anomaly exploration             │
└───────────────────────────────────────────────┘
```
---

## Evaluation matrice

1. ROC-AUC (Geo-Mean Fusion): 96.6%
2. PR-AUC (Geo-Mean Fusion): 84.2%
3. F1-Score: 76.5%
4. Precision: 70.0%
5. Recall: 84.3%

<img width="1280" height="800" alt="PHOTO-2026-05-13-23-26-15" src="https://github.com/user-attachments/assets/ed444b36-4df7-447c-bc80-27ecf1ddd6da" />

---
## 🛰️ Dataset

**NASA AVIRIS-NG (Airborne Visible/Infrared Imaging Spectrometer — Next Generation)**

| Property | Value |
|---|---|
| Source | NASA Jet Propulsion Laboratory |
| Processing Level | L2 (Surface Reflectance) |
| Spectral Range | 380 – 2510 nm |
| Spectral Bands | 425 contiguous bands |
| Spatial Resolution | ~5 m/pixel |
| Scene Location | Mississippi River Delta, Louisiana, USA |
| Access | [NASA Earthdata](https://search.earthdata.nasa.gov/search/granules?p=C2430019879-ORNL_CLOUD&pg[0][v]=f&pg[0][gsk]=-start_date&q=DeltaX_L2_AVIRIS_Reflectance_1988&ac=true) (free, registration required) |

> AVIRIS-NG L2 data is publicly available through NASA Earthdata. you can download from the link above (i have gave direct link but need registered account).


---

## ⚙️ Setup

### Prerequisites

- Python 3.10+
- 8 GB RAM minimum (16 GB recommended for full-scene processing)
- For cloud training: AWS SageMaker Studio Lab account (free)

### Installation

```bash
# Clone the repository
git clone https://github.com/hafan-961/Anomaly-Detection-in-Hyperspectral-Satellite-Imagery-for-Defence-Environmental-Surveillance.git
cd Anomaly-Detection-in-Hyperspectral-Satellite-Imagery-for-Defence-Environmental-Surveillance

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

tensorflow>=2.12.0
numpy>=1.24.0
scikit-learn>=1.3.0
spectral>=0.23.1
gdal>=3.6.0
flask>=2.3.0
folium>=0.14.0
matplotlib>=3.7.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0

---

### Configuration

Edit `config.yaml` to set your data paths and hyperparameters:

```yaml
data:
  scene_1: "data/raw/ang20150420t182808_rfl_v2p9/"
  scene_2: "data/raw/ang20150423t174831_rfl_v2p9/"
  pca_components: 30

model:
  latent_dim: 4
  epochs: 100
  batch_size: 512
  learning_rate: 0.001
  vae_beta: 0.5           # KL weight

scoring:
  threshold_precision_floor: 0.85

dashboard:
  port: 5000
  debug: false
```

---
## FUTURE SCOPE

#### This is not the final project , it needs many improvemnt for defence grade quality , need to train on more and updated dataset and need high computation power to make this happen. so we can take precution before any unfortunate happens.
---

## 👤 Author

**Muhammed Hafan**  
B.Tech Computer Science Engineering, Lovely Professional University (2023–2027)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-muhammed--hafan-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/muhammed-hafan)
[![LeetCode](https://img.shields.io/badge/LeetCode-Profile-FFA116?style=flat-square&logo=leetcode)](https://leetcode.com/u/ssYdyKjSfq/)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?style=flat-square&logo=github)](https://github.com/hafan-961)

---



<div align="center">

*Built for the intersection of deep learning and remote sensing — where every pixel tells a story.*

</div>



