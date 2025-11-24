import os
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import median_filter

# Path
BASE = "assets"
NOISE_DIR = os.path.join(BASE, "noise")

# Output directories
OUT_SMOOTH = os.path.join(BASE, "output_smoothing")
OUT_MEDIAN = os.path.join(BASE, "output_median")
OUT_ADAPTIVE = os.path.join(BASE, "output_adaptive")

os.makedirs(OUT_SMOOTH, exist_ok=True)
os.makedirs(OUT_MEDIAN, exist_ok=True)
os.makedirs(OUT_ADAPTIVE, exist_ok=True)

# -------- 1. Smoothing (mean filter) --------
def smoothing_filter(img_np):
    kernel = np.ones((5,5)) / 25.0
    padded = np.pad(img_np, ((2,2),(2,2)), mode='edge')
    out = np.zeros_like(img_np)

    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            region = padded[i:i+5, j:j+5]
            out[i,j] = np.sum(region * kernel)

    return out.astype(np.uint8)

# -------- 2. Median filter --------
def median_filter_np(img_np):
    out = median_filter(img_np, size=3)
    return out.astype(np.uint8)

# -------- 3. Adaptive (Lee) filter --------
def lee_filter(img_np, size=7):
    img = img_np.astype(np.float32)

    kernel = np.ones((size, size)) / (size * size)
    pad = size // 2

    # Local mean
    padded = np.pad(img, pad, mode='edge')
    local_mean = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+size, j:j+size]
            local_mean[i,j] = np.mean(region)

    # Local variance
    local_var = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+size, j:j+size]
            local_var[i,j] = np.var(region)

    # Noise variance estimation
    noise_var = np.mean(local_var)

    # Lee filter formula
    W = local_var / (local_var + noise_var)
    filtered = local_mean + W * (img - local_mean)

    return np.clip(filtered, 0, 255).astype(np.uint8)


# ============== MAIN LOOP ==============

for fname in os.listdir(NOISE_DIR):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(NOISE_DIR, fname)

        img = Image.open(path).convert("L")
        img_np = np.array(img).astype(np.float32)

        # --- Smoothing ---
        sm = smoothing_filter(img_np)
        Image.fromarray(sm).save(os.path.join(OUT_SMOOTH, f"smooth_{fname}"))

        # --- Median ---
        md = median_filter_np(img_np)
        Image.fromarray(md).save(os.path.join(OUT_MEDIAN, f"median_{fname}"))

        # --- Adaptive ---
        ad = lee_filter(img_np)
        Image.fromarray(ad).save(os.path.join(OUT_ADAPTIVE, f"adaptive_{fname}"))

print("Classical denoising complete.")
