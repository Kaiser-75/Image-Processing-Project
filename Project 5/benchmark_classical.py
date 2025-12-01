import os
import time
import json
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from metrics import mse_np, psnr_np, ssim_np

BASE = "assets"
CLEAN_DIR = os.path.join(BASE, "gray_bsd")     
NOISE_DIR = os.path.join(BASE, "noise_bsd")    
OUT_JSON = os.path.join(BASE, "benchmark_classical_results.json")

GAUSSIAN_NOISES = ["gauss15", "gauss25", "gauss50"]
SALT_PEPPER_NOISES = ["sp2", "sp5"]
ALL_NOISES = GAUSSIAN_NOISES + SALT_PEPPER_NOISES


def smoothing_filter(img_np):
    kernel = np.ones((5, 5)) / 25.0
    padded = np.pad(img_np, ((2, 2), (2, 2)), mode="edge")
    out = np.zeros_like(img_np)
    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            region = padded[i:i+5, j:j+5]
            out[i, j] = np.sum(region * kernel)
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
            local_mean[i, j] = np.mean(region)
            local_var[i, j] = np.var(region)

    noise_var = np.mean(local_var)
    W = local_var / (local_var + noise_var)
    filtered = local_mean + W * (img - local_mean)

    return np.clip(filtered, 0, 255).astype(np.uint8)


def compute(clean, out, runtime):
    return mse_np(clean, out), psnr_np(clean, out), ssim_np(clean, out), runtime * 1000.0


stats = {
    noise: {
        "smooth_mse": 0, "smooth_psnr": 0, "smooth_ssim": 0, "smooth_time": 0,
        "median_mse": 0, "median_psnr": 0, "median_ssim": 0, "median_time": 0,
        "adaptive_mse": 0, "adaptive_psnr": 0, "adaptive_ssim": 0, "adaptive_time": 0,
        "count": 0
    }
    for noise in ALL_NOISES
}


file_list = sorted([f for f in os.listdir(NOISE_DIR) if f.endswith(".png")])
print(f"Found {len(file_list)} noisy images.")

for fname in file_list:

    noise_path = os.path.join(NOISE_DIR, fname)
    noisy_np = np.array(Image.open(noise_path).convert("L"), dtype=np.float32)

    base = fname.split("_")[0]
    noise_label = fname.split("_")[1].replace(".png", "")

    if noise_label not in ALL_NOISES:
        print("Skipping unknown noise label:", fname)
        continue

    clean_path = os.path.join(CLEAN_DIR, base + ".png")
    if not os.path.exists(clean_path):
        print("Missing clean image for:", base)
        continue

    clean_np = np.array(Image.open(clean_path).convert("L"), dtype=np.float32)

    S = stats[noise_label]

    
    t0 = time.time()
    sm = smoothing_filter(noisy_np)
    t1 = time.time()
    mse, psnr, ssim, t = compute(clean_np, sm, t1 - t0)
    S["smooth_mse"] += mse
    S["smooth_psnr"] += psnr
    S["smooth_ssim"] += ssim
    S["smooth_time"] += t

 
    t0 = time.time()
    md = median_filter_custom(noisy_np)
    t1 = time.time()
    mse, psnr, ssim, t = compute(clean_np, md, t1 - t0)
    S["median_mse"] += mse
    S["median_psnr"] += psnr
    S["median_ssim"] += ssim
    S["median_time"] += t

    )
    t0 = time.time()
    ad = lee_filter(noisy_np)
    t1 = time.time()
    mse, psnr, ssim, t = compute(clean_np, ad, t1 - t0)
    S["adaptive_mse"] += mse
    S["adaptive_psnr"] += psnr
    S["adaptive_ssim"] += ssim
    S["adaptive_time"] += t

    S["count"] += 1

    print(f"[{noise_label}] {fname} processed.")


def avg(a, c): return a / c if c > 0 else None

final = {}

for noise, S in stats.items():
    c = S["count"]
    if c == 0:
        continue

    final[noise] = {
        "smoothing": {
            "psnr": avg(S["smooth_psnr"], c),
            "ssim": avg(S["smooth_ssim"], c),
            "mse": avg(S["smooth_mse"], c),
            "runtime_ms": avg(S["smooth_time"], c),
        },
        "median": {
            "psnr": avg(S["median_psnr"], c),
            "ssim": avg(S["median_ssim"], c),
            "mse": avg(S["median_mse"], c),
            "runtime_ms": avg(S["median_time"], c),
        },
        "adaptive": {
            "psnr": avg(S["adaptive_psnr"], c),
            "ssim": avg(S["adaptive_ssim"], c),
            "mse": avg(S["adaptive_mse"], c),
            "runtime_ms": avg(S["adaptive_time"], c),
        },
        "num_images": c,
    }

out_json = {
    "per_noise": final,
}

with open(OUT_JSON, "w") as f:
    json.dump(out_json, f, indent=4)

print("\n=== Classical BSD68 Benchmark Complete ===")
print("Saved results to:", OUT_JSON)
