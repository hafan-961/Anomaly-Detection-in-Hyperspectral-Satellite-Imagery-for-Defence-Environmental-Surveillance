import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import os


# 1. Loading the data
HDR = "../data/ang20210925t151637_rfl_v2z1.hdr"
DATA = "../data/ang20210925t151637_rfl_v2z1"

NO_DATA = -9999

print("Loading image...")
img = envi.open(HDR, DATA)

rows, cols, bands = img.shape
print(f"Rows:{rows}")
print(f"Cols:{cols}")
print(f"Bands:{bands}")
print(f"Total pixels: {rows * cols:,}")
print(f"Total values: {rows * cols * bands:,}")


# 2. Checks wavelength
wavelengths = np.array(img.bands.centers)
print(f"\nFirst 5 wavelengths: {wavelengths[:5]}")
print(f"Last  5 wavelengths: {wavelengths[-5:]}")
print(f"Wavelength range: {wavelengths[0]:.1f}nm → {wavelengths[-1]:.1f}nm")


# 3. loading a safe subset of the dataset
print("\nLoading first 500 rows...")
subset = img.read_subregion((0, 500), (0, cols))
subset = subset.astype(np.float32)
print(f"Subset shape: {subset.shape}")


# 4. Mask no data pixels
nodata_mask = np.any(subset <= NO_DATA, axis=-1)
valid_pixels_mask = ~nodata_mask

total_pixels = 500 * cols
valid_pixels = np.sum(valid_pixels_mask)
print(f"\nValid pixels:   {valid_pixels:,}")
print(f"No-data pixels: {total_pixels - valid_pixels:,}")
print(f"Valid %:        {valid_pixels / total_pixels * 100:.1f}%")


# 5 Basic stastistics on valid pixesl only
valid_data = subset[valid_pixels_mask]
print(f"\nMin reflectance:  {valid_data.min():.4f}")
print(f"Max reflectance:  {valid_data.max():.4f}")
print(f"Mean reflectance: {valid_data.mean():.4f}")
print(f"Std reflectance:  {valid_data.std():.4f}")


# 6 Rgb stretch function
def stretch(band):
    band = band.copy()
    band[band <= NO_DATA] = 0
    valid = band[band > 0]
    if len(valid) == 0:
        return band
    lo = np.percentile(valid, 2)
    hi = np.percentile(valid, 98)
    out = np.clip((band - lo) / (hi - lo + 1e-9), 0, 1)
    out[band <= 0] = 0
    return out


r = stretch(subset[:, :, 60])
g = stretch(subset[:, :, 35])
b = stretch(subset[:, :, 10])
rgb = np.stack([r, g, b], axis=-1)

# 7 NIR band
nir = stretch(subset[:, :, 100])

# 8 sample 3 valid pixels
valid_coords = np.argwhere(valid_pixels_mask)
sample1 = valid_coords[0]
sample2 = valid_coords[len(valid_coords) // 2]
sample3 = valid_coords[-1]

sig1 = subset[sample1[0], sample1[1], :]
sig2 = subset[sample2[0], sample2[1], :]
sig3 = subset[sample3[0], sample3[1], :]

# 9 mean spectrum
mean_spectrum = valid_data.mean(axis=0)
std_spectrum = valid_data.std(axis=0)


# 10 plot everything
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "AVIRIS-NG Delta-X — Exploratory Data Analysis", fontsize=16, fontweight="bold"
)

# Plot 1 — RGB preview
axes[0, 0].imshow(rgb)
axes[0, 0].set_title("RGB Preview (Bands 60, 35, 10)")
axes[0, 0].axis("off")

# Plot 2 — NIR Band
axes[0, 1].imshow(nir, cmap="RdYlGn")
axes[0, 1].set_title("NIR Band 100 (~880nm) — Vegetation")
axes[0, 1].axis("off")

# Plot 3 — Spectral Signatures
axes[1, 0].plot(
    wavelengths, sig1, label=f"Pixel {tuple(sample1)}", color="blue", linewidth=1.5
)
axes[1, 0].plot(
    wavelengths, sig2, label=f"Pixel {tuple(sample2)}", color="green", linewidth=1.5
)
axes[1, 0].plot(
    wavelengths, sig3, label=f"Pixel {tuple(sample3)}", color="red", linewidth=1.5
)
axes[1, 0].set_xlabel("Wavelength (nm)")
axes[1, 0].set_ylabel("Reflectance")
axes[1, 0].set_title("Spectral Signatures — 3 Sample Pixels")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot4  Mean spectrum ± std deviation
axes[1, 1].plot(wavelengths, mean_spectrum, color="black", linewidth=2, label="Mean")
axes[1, 1].fill_between(
    wavelengths,
    mean_spectrum - std_spectrum,
    mean_spectrum + std_spectrum,
    alpha=0.3,
    color="blue",
    label="±1 std",
)
axes[1, 1].set_xlabel("Wavelength (nm)")
axes[1, 1].set_ylabel("Reflectance")
axes[1, 1].set_title("Mean Spectral Signature ± Std Dev")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/01_eda.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved → outputs/01_eda.png")
print("Stage 2 Complete ✅")
