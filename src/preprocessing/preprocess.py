import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


HDR = "data/ang20210925t151637_rfl_v2z1.hdr"
DATA = "data/ang20210925t151637_rfl_v2z1"

NO_DATA = -9999
N_PCA_COMPONENTS = 30  # reduce 425 bands → 30 components

# Water absorption band ranges to remove (in nm)
# These are the spikes you saw in Stage 2
BAD_BAND_RANGES = [
    (1340, 1460),  # first water absorption region
    (1790, 1960),  # second water absorption region
    (2450, 2510),  # noisy end of sensor range
]


print("STAGE 3 — PREPROCESSING PIPELINE")
#2.loading data
print("\n[1/6] Loading data...")
img = envi.open(HDR, DATA)
wavelengths = np.array(img.bands.centers)
rows, cols, n_bands = img.shape
print(f"Shape: {rows} × {cols} × {n_bands}")

# Load subset for preprocessing
# Using first 1000 rows for speed — enough for training
print("\n[2/6] Loading 1000 rows into memory...")
subset = img.read_subregion((0, 1000), (0, cols))
subset = subset.astype(np.float32)
print(f"      Loaded: {subset.shape}")


# 3 removing no data pixels 
print("\n[3/6] Masking no-data pixels...")
valid_mask = ~np.any(subset <= NO_DATA, axis=-1)
print(f"Valid pixels:   {np.sum(valid_mask):,}")
print(f"Invalid pixels: {np.sum(~valid_mask):,}")


# 4 removing bad bands (water area)

print("\n[4/6] Removing water absorption bands...")

good_bands = np.ones(n_bands, dtype=bool)  # start with all bands good

for wl_min, wl_max in BAD_BAND_RANGES:
    bad = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    good_bands[bad] = False
    print(f"Removed {bad.sum()} bands in {wl_min}–{wl_max}nm range")

good_band_indices = np.where(good_bands)[0]
good_wavelengths = wavelengths[good_bands]
n_good_bands = good_band_indices.shape[0]
print(f"Bands before: {n_bands}")
print(f"Bands after:  {n_good_bands}")

# applyin band selection
subset_clean = subset[:, :, good_band_indices]
print(f"Clean cube shape: {subset_clean.shape}")


# 5 normalising valid pixels

print("\n[5/6] Normalizing...")

# Extract valid pixels only
valid_spectra = subset_clean[valid_mask]  # shape: (n_valid, n_good_bands)
print(f"Valid spectra shape: {valid_spectra.shape}")

# Clip any remaining outliers
valid_spectra = np.clip(valid_spectra, 0, 1)

# Per-band normalization (min-max using percentiles)
band_min = np.percentile(valid_spectra, 1, axis=0)
band_max = np.percentile(valid_spectra, 99, axis=0)
band_range = band_max - band_min
band_range[band_range == 0] = 1  # avoid division by zero

normalized = (valid_spectra - band_min) / band_range
normalized = np.clip(normalized, 0, 1)
print(f"Normalized min: {normalized.min():.4f}")
print(f"Normalized max: {normalized.max():.4f}")
print(f"Normalized mean: {normalized.mean():.4f}")


#6 PCA — DIMENSIONALITY REDUCTION
print(f"\n[6/6] PCA: {n_good_bands} bands → {N_PCA_COMPONENTS} components...")

pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
pca_result = pca.fit_transform(normalized)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)
print(f"Variance explained by {N_PCA_COMPONENTS} components: "
    f"{cumulative[-1]*100:.2f}%")

print(f"First 5 components explain: {cumulative[4]*100:.2f}%")
print(f"PCA output shape: {pca_result.shape}")


# 7  SAVE PREPROCESSED DATA
print("\nSaving preprocessed data...")
os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/valid_spectra_normalized.npy", normalized)
np.save("data/processed/pca_features.npy", pca_result)
np.save("data/processed/valid_mask.npy", valid_mask)
np.save("data/processed/good_wavelengths.npy", good_wavelengths)
np.save("data/processed/band_min.npy", band_min)
np.save("data/processed/band_max.npy", band_max)

print("Saved: data/processed/valid_spectra_normalized.npy")
print("Saved: data/processed/pca_features.npy")
print("Saved: data/processed/valid_mask.npy")
print("Saved: data/processed/good_wavelengths.npy")


# 8  PLOTING  PREPROCESSING RESULTS

print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("AVIRIS-NG Delta-X — Preprocessing Results", fontsize=16, fontweight="bold")

# Plot 1 — Good vs Bad bands
axes[0, 0].plot(wavelengths, good_bands.astype(float), color="green", linewidth=1.5)
for wl_min, wl_max in BAD_BAND_RANGES:
    axes[0, 0].axvspan(wl_min, wl_max, alpha=0.3, color="red", label="Removed")
axes[0, 0].set_title("Band Selection — Red = Removed Water Absorption Bands")
axes[0, 0].set_xlabel("Wavelength (nm)")
axes[0, 0].set_ylabel("Good Band (1=keep, 0=remove)")
axes[0, 0].grid(True, alpha=0.3)

# Plot 2 — Before vs After normalization (sample spectrum)
sample_raw = valid_spectra[1000]
sample_norm = normalized[1000]
axes[0, 1].plot(good_wavelengths, sample_raw, color="red", label="Before normalization", alpha=0.7)
axes[0, 1].plot(good_wavelengths, sample_norm, color="blue", label="After normalization", alpha=0.7)
axes[0, 1].set_title("Sample Spectrum — Before vs After Normalization")
axes[0, 1].set_xlabel("Wavelength (nm)")
axes[0, 1].set_ylabel("Reflectance")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3 — PCA Explained Variance
axes[1, 0].bar(range(1, N_PCA_COMPONENTS + 1), explained * 100, color="steelblue", alpha=0.7)
axes[1, 0].plot(
    range(1, N_PCA_COMPONENTS + 1),
    cumulative * 100,
    color="red",
    marker="o",
    markersize=3,
    label="Cumulative",)
axes[1, 0].axhline(y=95, color="green", linestyle="--", label="95% threshold")
axes[1, 0].set_title("PCA — Explained Variance per Component")
axes[1, 0].set_xlabel("Principal Component")
axes[1, 0].set_ylabel("Variance Explained (%)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4 — PCA Feature Space (first 2 components)
sample_idx = np.random.choice(pca_result.shape[0], 5000, replace=False)
axes[1, 1].scatter(
    pca_result[sample_idx, 0],
    pca_result[sample_idx, 1],
    alpha=0.1,
    s=1,
    color="steelblue",)
axes[1, 1].set_title("PCA Feature Space — PC1 vs PC2 (5000 random pixels)")
axes[1, 1].set_xlabel("Principal Component 1")
axes[1, 1].set_ylabel("Principal Component 2")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/02_preprocessing.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → outputs/02_preprocessing.png")


# 9. TRAIN / VAL / TEST SPLIT
print("\nCreating train/val/test split...")
n_samples = normalized.shape[0]
idx = np.random.permutation(n_samples)

train_end = int(0.70 * n_samples)
val_end = int(0.85 * n_samples)

train_idx = idx[:train_end]
val_idx = idx[train_end:val_end]
test_idx = idx[val_end:]

X_train = normalized[train_idx]
X_val = normalized[val_idx]
X_test = normalized[test_idx]

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_val.npy", X_val)
np.save("data/processed/X_test.npy", X_test)

print(f"Train: {X_train.shape}")
print(f"Val:{X_val.shape}")
print(f"Test:{X_test.shape}")


print("\nFiles saved in data/processed/:")
print(" valid_spectra_normalized.npy  ← full normalized dataset")
print("pca_features.npy← PCA reduced features")
print(" valid_mask.npy ← which pixels are valid")
print("good_wavelengths.npy ← cleaned wavelength list")
print("X_train.npy  ← training set (70%)")
print("X_val.npy ← validation set (15%)")
print(" X_test.npy ← test set (15%)")
