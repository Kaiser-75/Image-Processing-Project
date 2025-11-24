import os
import time
import json
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

from metrics import mse_np, psnr_np, ssim_np

BASE = r"assets"
NOISE_DIR = os.path.join(BASE, "noise")

OUT_SMOOTH = os.path.join(BASE, "output_smoothing")
OUT_MEDIAN = os.path.join(BASE, "output_median")
OUT_ADAPTIVE = os.path.join(BASE, "output_adaptive")
OUT_JSON = os.path.join(BASE, "metrics")

os.makedirs(OUT_SMOOTH, exist_ok=True)
os.makedirs(OUT_MEDIAN, exist_ok=True)
os.makedirs(OUT_ADAPTIVE, exist_ok=True)
os.makedirs(OUT_JSON, exist_ok=True)

def smoothing_filter(img_np):
    kernel = np.ones((5,5)) / 25.0
    padded = np.pad(img_np, ((2,2),(2,2)), mode='edge')
    out = np.zeros_like(img_np)
    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            region = padded[i:i+5, j:j+5]
            out[i,j] = np.sum(region * kernel)
    return out.astype(np.uint8)

def median_filter_custom(img_np):
    out = median_filter(img_np, size=3)
    return out.astype(np.uint8)

def lee_filter(img_np, size=7):
    img = img_np.astype(np.float32)
    pad = size // 2
    padded = np.pad(img, pad, mode="edge")

    local_mean = np.zeros_like(img)
    local_var = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+size, j:j+size]
            local_mean[i,j] = np.mean(region)
            local_var[i,j] = np.var(region)

    noise_var = np.mean(local_var)
    W = local_var / (local_var + noise_var)
    filtered = local_mean + W * (img - local_mean)

    return np.clip(filtered, 0, 255).astype(np.uint8)

def compute_metrics(clean, out, runtime):
    return {
        "mse": mse_np(clean, out),
        "psnr": psnr_np(clean, out),
        "ssim": ssim_np(clean, out),
        "runtime_ms": round(runtime * 1000, 3)
    }

def find_clean_image(base_name):
    candidates = [
        os.path.join(BASE, base_name + ".png"),
        os.path.join(BASE, base_name + ".jpg"),
        os.path.join(BASE, base_name + ".jpeg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

results = {
    "lenna": {},
    "nature": {},
    "potrait": {},
    "street": {}
}

for fname in os.listdir(NOISE_DIR):
    if not fname.lower().endswith(".png"):
        continue

    base_name = fname.split("_")[0]
    noise_name = fname.replace(base_name + "_", "").replace(".png","")

    if base_name not in results:
        continue

    clean_path = find_clean_image(base_name)
    if clean_path is None:
        print("Missing clean image for:", base_name)
        continue

    noisy_np = np.array(Image.open(os.path.join(NOISE_DIR, fname)).convert("L")).astype(np.float32)
    clean_np = np.array(Image.open(clean_path).convert("L")).astype(np.float32)

    sample_metrics = {}

    t0 = time.time()
    sm = smoothing_filter(noisy_np)
    t1 = time.time()
    Image.fromarray(sm).save(os.path.join(OUT_SMOOTH, "smooth_" + fname))
    sample_metrics["smoothing"] = compute_metrics(clean_np, sm, t1 - t0)

    t0 = time.time()
    md = median_filter_custom(noisy_np)
    t1 = time.time()
    Image.fromarray(md).save(os.path.join(OUT_MEDIAN, "median_" + fname))
    sample_metrics["median"] = compute_metrics(clean_np, md, t1 - t0)

    t0 = time.time()
    ad = lee_filter(noisy_np)
    t1 = time.time()
    Image.fromarray(ad).save(os.path.join(OUT_ADAPTIVE, "adaptive_" + fname))
    sample_metrics["adaptive"] = compute_metrics(clean_np, ad, t1 - t0)

    results[base_name][noise_name] = sample_metrics

    print("Processed", fname)

for base_name, data in results.items():
    out_path = os.path.join(OUT_JSON, base_name + ".json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)

print("Saved grouped JSON files.")
