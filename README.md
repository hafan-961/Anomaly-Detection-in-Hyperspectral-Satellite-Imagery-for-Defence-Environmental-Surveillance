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


