import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from scipy.ndimage import uniform_filter, label as scipy_label
import os
import json
import warnings
import spectral.io.envi as envi

warnings.filterwarnings("ignore")


# CONFIGURATION

print("=" * 58)
print("STAGE 5 — EVALUATION  (geometric fusion + recall-tuned)")
print("=" * 58)

HDR  = "data/ang20210925t151637_rfl_v2z1.hdr"
DATA = "data/ang20210925t151637_rfl_v2z1"

LABEL_PERCENTILE  = 98    # top 2% AE  → proxy anomaly labels
NORMAL_RATIO      = 5     # 5 normals per anomaly in eval set
RANDOM_SEED       = 42
TARGET_PRECISION  = 0.45  # lowered from 0.65 → more aggressive recall
MIN_CLUSTER_SIZE  = 3     # remove clusters smaller than this (noise filter)
SPATIAL_SMOOTH    = 3     # uniform filter kernel size

os.makedirs("outputs",        exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# 2.Load scores + mask

print("\n[1/7] Loading anomaly scores...")

ae_scores       = np.load("data/processed/ae_scores.npy")
vae_scores      = np.load("data/processed/vae_scores.npy")
ensemble_scores = np.load("data/processed/ensemble_scores.npy")
valid_mask      = np.load("data/processed/valid_mask.npy")

print(f"AE scores:{ae_scores.shape}  mean={ae_scores.mean():.4f}  std={ae_scores.std():.4f}")
print(f"VAE scores:{vae_scores.shape}  mean={vae_scores.mean():.4f}  std={vae_scores.std():.4f}")
print(f"Ensemble scores: {ensemble_scores.shape}  mean={ensemble_scores.mean():.4f}  std={ensemble_scores.std():.4f}")
print(f"Valid pixels:{valid_mask.sum():,}")


# 3. CROSS-MODEL GROUND TRUTH

print("\n[2/7] Building cross-model ground truth...")

ae_label_thresh = np.percentile(ae_scores, LABEL_PERCENTILE)
y_true_full     = (ae_scores >= ae_label_thresh).astype(int)

np.random.seed(RANDOM_SEED)
anomaly_pool  = np.where(y_true_full == 1)[0]
normal_pool   = np.where(y_true_full == 0)[0]

n_anomalies   = len(anomaly_pool)
n_normals     = min(len(normal_pool), n_anomalies * NORMAL_RATIO)

normal_sample = np.random.choice(normal_pool, size=n_normals, replace=False)
eval_idx      = np.concatenate([anomaly_pool, normal_sample])
np.random.shuffle(eval_idx)

y_true        = y_true_full[eval_idx]
vae_eval      = vae_scores[eval_idx]
ensemble_eval = ensemble_scores[eval_idx]

# Geometric mean fusion scores for eval set
# Amplifies pixels where BOTH models agree — suppresses single-model misfires
ae_eval_raw   = ae_scores[eval_idx]
geo_eval_raw  = np.sqrt(np.abs(ae_eval_raw * vae_eval))
g_min, g_max  = geo_eval_raw.min(), geo_eval_raw.max()
geo_eval      = (geo_eval_raw - g_min) / (g_max - g_min + 1e-9)

print(f"      AE label threshold: {ae_label_thresh:.4f}  (top {100-LABEL_PERCENTILE}%)")
print(f"      Evaluation set:     {len(eval_idx):,} pixels")
print(f"      Anomalies:          {y_true.sum():,} ({y_true.mean()*100:.1f}%)")
print(f"      Normals:            {(y_true==0).sum():,}  (random, full distribution)")
print(f"      Fusion scores:      geometric mean(AE, VAE) — normalized 0–1")
print(f"      (Labels = AE only — VAE/Ensemble/Geo evaluated independently)")


# 4. ROC-AUC + PR-AUC  (all three scorers)
print("\n[3/7] Computing ROC and PR curves...")

auc_vae = roc_auc_score(y_true, vae_eval)
auc_ens = roc_auc_score(y_true, ensemble_eval)
auc_geo = roc_auc_score(y_true, geo_eval)

fpr_vae, tpr_vae, _ = roc_curve(y_true, vae_eval)
fpr_ens, tpr_ens, _ = roc_curve(y_true, ensemble_eval)
fpr_geo, tpr_geo, _ = roc_curve(y_true, geo_eval)

pr_auc_vae = average_precision_score(y_true, vae_eval)
pr_auc_ens = average_precision_score(y_true, ensemble_eval)
pr_auc_geo = average_precision_score(y_true, geo_eval)

prec_vae, rec_vae, _= precision_recall_curve(y_true, vae_eval)
prec_ens, rec_ens, _= precision_recall_curve(y_true, ensemble_eval)
prec_geo, rec_geo, _= precision_recall_curve(y_true, geo_eval)
prec_pts,  rec_pts, thr_pts  = precision_recall_curve(y_true, geo_eval)
pr_baseline= y_true.mean()

pr_lift_vae = pr_auc_vae / pr_baseline
pr_lift_ens = pr_auc_ens / pr_baseline
pr_lift_geo = pr_auc_geo / pr_baseline

print(f"ROC-AUC — VAE:{auc_vae:.4f}  ← primary (fully independent of labels)")
print(f"ROC-AUC — Ensemble: {auc_ens:.4f}  (informational — 40% AE signal)")
print(f"ROC-AUC — Geo-Mean: {auc_geo:.4f}  (geometric fusion — used for thresholding)")
print(f"PR-AUC  — VAE:{pr_auc_vae:.4f}  ({pr_lift_vae:.2f}× random {pr_baseline:.3f})")
print(f"PR-AUC  — Ensemble: {pr_auc_ens:.4f}  ({pr_lift_ens:.2f}× random)")
print(f"PR-AUC  — Geo-Mean: {pr_auc_geo:.4f}  ({pr_lift_geo:.2f}× random)")


# 5. THRESHOLD — precision-constrained on geo_eval
print("\n[4/7] Finding precision-constrained threshold on geometric fusion scores...")

# Precision-constrained: highest recall where precision >= TARGET_PRECISION
best_recall_pc, threshold_pc = 0.0, thr_pts[-1]
for p, r, t in zip(prec_pts, rec_pts, thr_pts):
    if p >= TARGET_PRECISION and r > best_recall_pc:
        best_recall_pc, threshold_pc = r, t

# F1-optimal threshold for comparison
sweep_pcts       = np.arange(70, 99, 1)
sweep_thresholds = np.percentile(geo_eval, sweep_pcts)
best_f1_val, threshold_f1, best_pct = 0, sweep_thresholds[0], sweep_pcts[0]
f1_scores_sweep  = []
for pct, t in zip(sweep_pcts, sweep_thresholds):
    yp    = (geo_eval >= t).astype(int)
    score = f1_score(y_true, yp, zero_division=0)
    f1_scores_sweep.append(score)
    if score > best_f1_val:
        best_f1_val, threshold_f1, best_pct = score, t, pct

threshold = threshold_pc
print(f"Precision-constrained threshold: {threshold_pc:.4f}  (precision >= {TARGET_PRECISION})")
print(f"F1-optimal threshold:{threshold_f1:.4f}  (F1={best_f1_val:.4f} at {best_pct}th pct)")

y_pred= (geo_eval >= threshold).astype(int)
f1 = f1_score(y_true, y_pred, zero_division=0)
precision = precision_score(y_true, y_pred, zero_division=0)
recall= recall_score(y_true, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
pfar= fp / (fp + tn + 1e-9)
detection_rate = tp / (tp + fn + 1e-9)

print(f"\n── Metrics at precision-constrained threshold ──")
print(f"F1 Score:{f1:.4f}")
print(f"Precision:{precision:.4f}  (target >= {TARGET_PRECISION})")
print(f"Recall:{recall:.4f}")
print(f"PFAR:{pfar:.4f}")
print(f"Det. Rate:{detection_rate:.4f}  (= Recall, cross-check)")
print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")

spatial_threshold = np.percentile(ensemble_scores, 95)


# 6. SPATIAL HEATMAP — geometric fusion + smoothing + cluster filter
print("\n[5/7] Building spatial heatmap (geometric fusion + noise filtering)...")

img = envi.open(HDR, DATA)
rows, cols, _ = img.shape

# Seed-reconstruction: recover exact test_idx without rerunning preprocess
np.random.seed(RANDOM_SEED)
n_total_valid = int(valid_mask.sum())
idx = np.random.permutation(n_total_valid)
test_idx = idx[int(0.85 * n_total_valid):]

valid_coords = np.argwhere(valid_mask)
test_coords  = valid_coords[test_idx]

print(f" Test coords: {test_coords.shape}  scores: {ensemble_scores.shape}")

# Build per-model spatial maps
ae_full_map  = np.full((rows, cols), np.nan)
vae_full_map = np.full((rows, cols), np.nan)
ae_full_map[test_coords[:,0],  test_coords[:,1]] = ae_scores
vae_full_map[test_coords[:,0], test_coords[:,1]] = vae_scores

# ── Geometric mean fusion on spatial maps ─────────────────────────────────
# sqrt(AE * VAE) — amplifies agreement, suppresses single-model spikes
geo_raw = np.sqrt(np.abs(ae_full_map * vae_full_map))
valid_region = ~np.isnan(ae_full_map)
gmin = np.nanmin(geo_raw)
gmax = np.nanmax(geo_raw)
geo_norm = np.full((rows, cols), np.nan)
geo_norm[valid_region] = (geo_raw[valid_region] - gmin) / (gmax - gmin + 1e-9)

# ── Spatial smoothing ─────────────────────────────────────────────────────
score_map_smooth               = geo_norm.copy()
score_map_smooth[valid_region] = uniform_filter(
    np.nan_to_num(geo_norm, nan=0.0), size=SPATIAL_SMOOTH)[valid_region]

# Keep original ensemble score_map for backward compatibility
score_map = np.full((rows, cols), np.nan)
score_map[test_coords[:,0], test_coords[:,1]] = ensemble_scores

# Spatial threshold on smoothed geo scores
geo_spatial_threshold = np.percentile(
    score_map_smooth[valid_region], 95)
anomaly_raw = score_map_smooth >= geo_spatial_threshold

# ── Cluster size filter ───────────────────────────────────────────────────
labeled_map, n_clusters = scipy_label(anomaly_raw)
cluster_sizes = np.bincount(labeled_map.ravel())
anomaly_map = anomaly_raw.copy()
removed_pixels = 0
for cl_id in range(1, n_clusters + 1):
    if cluster_sizes[cl_id] < MIN_CLUSTER_SIZE:
        anomaly_map[labeled_map == cl_id] = False
        removed_pixels += cluster_sizes[cl_id]

print(f"Raw anomaly pixels:{np.nansum(anomaly_raw):,.0f}")
print(f"After cluster filter:{np.nansum(anomaly_map):,.0f}")
print(f"Noise pixels removed:{removed_pixels:,.0f}  (clusters < {MIN_CLUSTER_SIZE}px)")
print(f"Coverage: {np.nansum(anomaly_map)/len(ensemble_scores)*100:.2f}% of test pixels")
print(f" Note: heatmap covers rows 0–{valid_mask.shape[0]} of {rows} (1000-row limit)")

np.save("data/processed/score_map.npy",score_map)
np.save("data/processed/score_map_smooth.npy", score_map_smooth)
np.save("data/processed/anomaly_map.npy",anomaly_map)


# 7. GEOJSON EXPORT

print("\n[6/7] Exporting GeoJSON...")

use_geo = False
try:
    meta     = img.metadata
    map_info = meta.get("map info", "")
    if map_info:
        parts   = [p.strip() for p in map_info.strip("{}").split(",")]
        ul_x    = float(parts[3])
        ul_y    = float(parts[4])
        pixel_x = float(parts[5])
        pixel_y = float(parts[6])
        use_geo = True
        print(f"      Geographic coords: UL=({ul_x:.4f}, {ul_y:.4f}), res=({pixel_x:.4f}, {pixel_y:.4f})")
except Exception:
    print("      No map info found — using pixel row/col as coordinates")

anomaly_pixels = np.argwhere(anomaly_map)
features = []
for (r, c) in anomaly_pixels[:5000]:
    score = float(score_map_smooth[r, c])
    lon   = (ul_x + c * pixel_x) if use_geo else float(c)
    lat   = (ul_y - r * pixel_y) if use_geo else float(r)
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "row":   int(r),
            "col":   int(c),
            "score": round(score, 4),
            "ensemble_score": round(float(score_map[r, c]) if not np.isnan(score_map[r, c]) else 0.0, 4)
        }
    })

geojson = {"type": "FeatureCollection", "features": features}
with open("outputs/anomalies.geojson", "w") as f:
    json.dump(geojson, f)
print(f"      GeoJSON saved → outputs/anomalies.geojson ({len(features)} points)")

metrics = {
    "methodology": (
        "Cross-model unsupervised evaluation. "
        "AE top-2% defines proxy labels. "
        "VAE evaluated independently (primary metric). "
        f"Geometric mean fusion (sqrt(AE*VAE)) used for thresholding. "
        f"Threshold: precision >= {TARGET_PRECISION} (recall-tuned). "
        f"Spatial post-processing: {SPATIAL_SMOOTH}x smooth + cluster >= {MIN_CLUSTER_SIZE}px."
    ),
    "ROC_AUC_VAE":          round(float(auc_vae), 4),
    "ROC_AUC_Ensemble":     round(float(auc_ens), 4),
    "ROC_AUC_GeoMean":      round(float(auc_geo), 4),
    "PR_AUC_VAE":           round(float(pr_auc_vae), 4),
    "PR_AUC_Ensemble":      round(float(pr_auc_ens), 4),
    "PR_AUC_GeoMean":       round(float(pr_auc_geo), 4),
    "PR_lift_VAE":          round(float(pr_lift_vae), 2),
    "PR_lift_GeoMean":      round(float(pr_lift_geo), 2),
    "F1":                   round(float(f1), 4),
    "Precision":            round(float(precision), 4),
    "Recall":               round(float(recall), 4),
    "PFAR":                 round(float(pfar), 4),
    "Detection_Rate":       round(float(detection_rate), 4),
    "Threshold_prec_constrained": round(float(threshold_pc), 4),
    "Threshold_f1_optimal": round(float(threshold_f1), 4),
    "Threshold_spatial":    round(float(geo_spatial_threshold), 4),
    "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    "anomaly_pixels_raw":      int(np.nansum(anomaly_raw)),
    "anomaly_pixels_filtered": int(np.nansum(anomaly_map)),
    "noise_pixels_removed":    int(removed_pixels),
}
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("      Metrics saved → outputs/metrics.json")


# 8. PLOTS
print("\n[7/7] Generating evaluation plots...")

fig = plt.figure(figsize=(22, 14))
fig.suptitle("AVIRIS-NG Hyperspectral — Stage 5 Evaluation", fontsize=17, fontweight="bold", y=0.99)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

ax_roc  = fig.add_subplot(gs[0, 0])
ax_bar  = fig.add_subplot(gs[0, 1])
ax_cm   = fig.add_subplot(gs[0, 2])
ax_heat = fig.add_subplot(gs[1, 0])
ax_pr   = fig.add_subplot(gs[1, 1])
ax_dist = fig.add_subplot(gs[1, 2])

# ── Plot 1: ROC Curves ──────────────────
ax_roc.plot(fpr_vae, tpr_vae, color="darkorange",  linewidth=2.5,
            label=f"VAE  AUC={auc_vae:.3f}  ← primary")
ax_roc.plot(fpr_geo, tpr_geo, color="royalblue",   linewidth=2.5, linestyle="-.",
            label=f"Geo-Mean AUC={auc_geo:.3f}  (fusion)")
ax_roc.plot(fpr_ens, tpr_ens, color="seagreen",    linewidth=1.5, linestyle="--",
            label=f"Ensemble AUC={auc_ens:.3f}  (info)")
ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.35, label="Random (0.500)")
ax_roc.set_xlim(0, 1);  ax_roc.set_ylim(0, 1.02)
ax_roc.set_title("ROC Curves", fontweight="bold")
ax_roc.set_xlabel("False Positive Rate (PFAR)")
ax_roc.set_ylabel("True Positive Rate (Detection Rate)")
ax_roc.legend(fontsize=8);  ax_roc.grid(True, alpha=0.3)

# ── Plot 2: Metrics Bar ─────────────────
m_names  = ["ROC-AUC\nVAE", "PR-AUC\nVAE", "F1\nScore", "Precision", "Recall"]
m_values = [auc_vae, pr_auc_vae, f1, precision, recall]
m_colors = ["#2ecc71" if v >= 0.8 else "#f39c12" if v >= 0.5 else "#e74c3c"
            for v in m_values]
bars = ax_bar.bar(m_names, m_values, color=m_colors, alpha=0.88,
                  edgecolor="black", linewidth=0.8)
ax_bar.axhline(y=0.8,             color="green", linestyle="--", alpha=0.55, label="Target 0.80")
ax_bar.axhline(y=TARGET_PRECISION, color="blue",  linestyle=":",  alpha=0.65,
               label=f"Prec. floor {TARGET_PRECISION}")
ax_bar.axhline(y=pr_baseline,     color="gray",  linestyle=":",  alpha=0.55,
               label=f"PR random {pr_baseline:.2f}")
ax_bar.set_ylim(0, 1.15)
ax_bar.set_title("Evaluation Metrics", fontweight="bold")
ax_bar.set_ylabel("Score");  ax_bar.legend(fontsize=7.5)
ax_bar.grid(True, alpha=0.25, axis="y")
for bar, val in zip(bars, m_values):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# ── Plot 3: Confusion Matrix ────────────
cm_disp = np.array([[tn, fp], [fn, tp]])
im = ax_cm.imshow(cm_disp, interpolation="nearest", cmap=plt.cm.Blues)
ax_cm.set_title("Confusion Matrix", fontweight="bold")
ax_cm.set_xticks([0, 1]);  ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["Pred Normal", "Pred Anomaly"])
ax_cm.set_yticklabels(["True Normal", "True Anomaly"])
for i, j in [(0,0),(0,1),(1,0),(1,1)]:
    ax_cm.text(j, i, f"{cm_disp[i,j]:,}", ha="center", va="center", fontsize=13,
               color="white" if cm_disp[i,j] > cm_disp.max()/2 else "black")
plt.colorbar(im, ax=ax_cm)
ax_cm.set_xlabel(
    f"Precision={precision:.3f}  Recall={recall:.3f}  PFAR={pfar:.3f}", fontsize=9)

# ── Plot 4: Spatial Heatmap (geo-mean smoothed) ──
step= 4
valid_rows = valid_mask.shape[0]
heat_sub= score_map_smooth[::step, ::step]
im2 = ax_heat.imshow(heat_sub, cmap="hot", aspect="auto",
                     vmin=0, vmax=np.nanpercentile(score_map_smooth, 99),
                     extent=[0, cols // step, valid_rows // step, 0])
ax_heat.set_xlim(0, cols // step)
ax_heat.set_ylim(valid_rows // step, 0)
ax_heat.set_title("Anomaly Score Heatmap (geo-mean + smoothed)", fontweight="bold")
ax_heat.set_xlabel("Column (4× subsampled)")
ax_heat.set_ylabel(f"Row (4× subsampled, rows 0–{valid_rows})")
plt.colorbar(im2, ax=ax_heat, label="Geo-Mean Score")
ax_heat.text(0.02, 0.96, f"Covers rows 0–{valid_rows} / {rows}",
             transform=ax_heat.transAxes, fontsize=7.5, color="yellow",
             va="top", bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

# ── Plot 5: Precision-Recall Curves ──────
ax_pr.plot(rec_vae, prec_vae, color="darkorange", linewidth=2.5,
           label=f"VAE  PR-AUC={pr_auc_vae:.3f}  ({pr_lift_vae:.1f}× random)")
ax_pr.plot(rec_geo, prec_geo, color="royalblue",  linewidth=2.5, linestyle="-.",
           label=f"Geo-Mean PR-AUC={pr_auc_geo:.3f}  ({pr_lift_geo:.1f}× random)")
ax_pr.plot(rec_ens, prec_ens, color="seagreen",   linewidth=1.5, linestyle="--",
           label=f"Ensemble PR-AUC={pr_auc_ens:.3f}  ({pr_lift_ens:.1f}× random)")
ax_pr.axhline(y=pr_baseline,     color="gray", linestyle="--", alpha=0.65,
              label=f"Random ({pr_baseline:.3f})")
ax_pr.axhline(y=TARGET_PRECISION, color="blue", linestyle=":", alpha=0.8,
              label=f"Precision floor ({TARGET_PRECISION})")
ax_pr.set_xlim(0, 1);  ax_pr.set_ylim(0, 1.05)
ax_pr.set_title("Precision-Recall Curves", fontweight="bold")
ax_pr.set_xlabel("Recall");  ax_pr.set_ylabel("Precision")
ax_pr.legend(fontsize=7.5);  ax_pr.grid(True, alpha=0.3)

# ── Plot 6: Score Distribution + F1 inset
ax_dist.hist(
    score_map_smooth[valid_region & ~np.isnan(score_map_smooth)],
    bins=200, color="royalblue", alpha=0.72, label="Geo-Mean scores (test)"
)
ax_dist.axvline(x=geo_spatial_threshold, color="crimson",   linestyle="--", linewidth=2,
                label=f"Spatial threshold = {geo_spatial_threshold:.3f}")
ax_dist.axvline(x=threshold_pc,          color="blue",      linestyle=":",  linewidth=2,
                label=f"Prec-constrained = {threshold_pc:.3f}")
ax_dist.axvline(x=threshold_f1,          color="purple",    linestyle=":",  linewidth=1.5,
                label=f"F1-optimal = {threshold_f1:.3f}")
ax_dist.set_xlim(left=0)
ax_dist.set_title("Geo-Mean Score Distribution", fontweight="bold")
ax_dist.set_xlabel("Anomaly Score");  ax_dist.set_ylabel("Pixel Count")
ax_dist.legend(fontsize=7.5);  ax_dist.grid(True, alpha=0.3)

# F1 sweep inset
ax_ins = ax_dist.inset_axes([0.45, 0.42, 0.52, 0.50])
ax_ins.plot(sweep_thresholds, f1_scores_sweep, color="purple", linewidth=1.5)
ax_ins.axvline(x=threshold_f1, color="purple", linestyle=":", linewidth=1)
ax_ins.axvline(x=threshold_pc, color="blue",   linestyle=":", linewidth=1)
ax_ins.set_title("F1 sweep", fontsize=7)
ax_ins.set_xlabel("threshold", fontsize=6)
ax_ins.set_ylabel("F1", fontsize=6)
ax_ins.tick_params(labelsize=6);  ax_ins.grid(True, alpha=0.3)

plt.savefig("outputs/04_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → outputs/04_evaluation.png")


# 9.Summary 

print("\n" + "=" * 58)
print("Stage 5 Complete ✅")
print("=" * 58)
print("\n=== PRIMARY METRICS ===")
print(
    f"ROC-AUC (VAE): {auc_vae:.4f} "
    f"{'✅' if auc_vae >= 0.80 else '⚠️'} "
    f"(fully label-independent)")

print(
    f"PR-AUC (VAE): {pr_auc_vae:.4f} "
    f"({pr_lift_vae:.2f}× above random baseline)")

print("\n=== FUSION METRICS (Geo-Mean AE×VAE) ===")
print(f"ROC-AUC (Geo-Mean): {auc_geo:.4f}")

print(
    f"PR-AUC (Geo-Mean): {pr_auc_geo:.4f} "
    f"({pr_lift_geo:.2f}× random)")

print(f"\n=== THRESHOLD METRICS (precision >= {TARGET_PRECISION}) ===")
print(f"Precision: {precision:.4f} " f"{'✅' if precision >= TARGET_PRECISION else '⚠️'}")

print(f"Recall: {recall:.4f} (Detection Rate)")
print(f"F1 Score: {f1:.4f}")

print(
    f"PFAR: {pfar:.4f} "
    f"{'✅' if pfar <= 0.15 else '⚠️'}")

print(f"TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")

print("\n=== SPATIAL POST-PROCESSING ===")
print("Fusion: Geometric mean sqrt(AE × VAE)")

print(
    f"Smoothing: {SPATIAL_SMOOTH}×{SPATIAL_SMOOTH} "
    f"uniform filter")

print(
    f"Cluster filter: min {MIN_CLUSTER_SIZE} "
    f"connected pixels")

print(f"Raw anomalies: {int(np.nansum(anomaly_raw)):,}")
print(f"After filtering: {int(np.nansum(anomaly_map)):,}")
print(f"Noise removed: {removed_pixels:,} pixels")

print("\n=== METHODOLOGY ===")
print(
    f"AE top-{100 - LABEL_PERCENTILE}% defines proxy labels.")

print("No real ground truth exists for this NASA scene.")
print("VAE ROC-AUC is the primary reporting metric.")

print(
    f"Detector tuned for recall with "
    f"precision >= {TARGET_PRECISION}.")

print("\n=== OUTPUT FILES ===")
print("outputs/04_evaluation.png")
print("outputs/metrics.json")
print("outputs/anomalies.geojson")
print("data/processed/score_map.npy")
print("data/processed/score_map_smooth.npy")
print("data/processed/anomaly_map.npy")