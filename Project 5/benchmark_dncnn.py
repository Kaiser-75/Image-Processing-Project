import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from metrics import mse_np, psnr_np, ssim_np


BASE = "assets"

CLEAN_DIR = os.path.join(BASE, "gray_bsd")   
NOISE_DIR = os.path.join(BASE, "noise_bsd")  

OUT_JSON = os.path.join(BASE, "benchmark_dncnn_results.json")
MODEL_PATH = os.path.join(BASE, "weights", "dncnn_gray_blind.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

GAUSSIAN_NOISES = ["gauss15", "gauss25", "gauss50"]
SALT_PEPPER_NOISES = ["sp2", "sp5"]
ALL_NOISES = GAUSSIAN_NOISES + SALT_PEPPER_NOISES



class DnCNN(nn.Module):
    def __init__(self, num_layers=20, nc=64):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(1, nc, 3, 1, 1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(nc, nc, 3, 1, 1))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(nc, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.model(x)
        return x - noise



print("Loading DnCNN weights from:", MODEL_PATH)
state = torch.load(MODEL_PATH, map_location="cpu")


first_key = list(state.keys())[0]
if not first_key.startswith("model."):
    state = {"model." + k: v for k, v in state.items()}

model = DnCNN(num_layers=20).to(device)
model.load_state_dict(state, strict=False)
model.eval()
print("DnCNN loaded.")



def load_clean_gray(base_name: str) -> np.ndarray:
    """
    Load clean grayscale image from gray_bsd.
    Expects '<base_name>.png' in CLEAN_DIR.
    """
    path = os.path.join(CLEAN_DIR, base_name + ".png")
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def dncnn_denoise(noisy_np: np.ndarray):
    """
    noisy_np: H x W, float32 in [0, 255]
    returns: (denoised_uint8, runtime_ms)
    """
    x = noisy_np / 255.0
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model(x).cpu().squeeze().numpy()
    t1 = time.time()

    den = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    runtime_ms = (t1 - t0) * 1000.0
    return den, runtime_ms


stats = {
    noise_type: {
        "mse_sum": 0.0,
        "psnr_sum": 0.0,
        "ssim_sum": 0.0,
        "time_sum": 0.0,
        "count": 0
    }
    for noise_type in ALL_NOISES
}



file_list = sorted(
    [f for f in os.listdir(NOISE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
)

print(f"Found {len(file_list)} noisy images in {NOISE_DIR}")

for fname in file_list:
    noisy_path = os.path.join(NOISE_DIR, fname)
    noisy_img = Image.open(noisy_path).convert("L")
    noisy_np = np.array(noisy_img, dtype=np.float32)


    base_part = os.path.splitext(fname)[0]
    if "_" not in base_part:
        print("Skipping (unexpected name format):", fname)
        continue

    base_name, noise_label = base_part.split("_", 1)

    if noise_label not in ALL_NOISES:
        print("Skipping (unknown noise label):", fname)
        continue

    clean_np = load_clean_gray(base_name)
    if clean_np is None:
        print("Missing clean image for", base_name)
        continue

    den, runtime_ms = dncnn_denoise(noisy_np)

    mse = mse_np(clean_np, den)
    psnr = psnr_np(clean_np, den)
    ssim = ssim_np(clean_np, den)

    s = stats[noise_label]
    s["mse_sum"] += mse
    s["psnr_sum"] += psnr
    s["ssim_sum"] += ssim
    s["time_sum"] += runtime_ms
    s["count"] += 1

    print(
        f"[{noise_label}] {fname} -> "
        f"PSNR={psnr:.2f}, SSIM={ssim:.4f}, Time={runtime_ms:.2f}ms"
    )



def avg_or_none(sum_val, count):
    return (sum_val / count) if count > 0 else None


results_per_noise = {}
for noise_label, s in stats.items():
    c = s["count"]
    if c == 0:
        continue
    results_per_noise[noise_label] = {
        "avg_mse": avg_or_none(s["mse_sum"], c),
        "avg_psnr": avg_or_none(s["psnr_sum"], c),
        "avg_ssim": avg_or_none(s["ssim_sum"], c),
        "avg_runtime_ms": avg_or_none(s["time_sum"], c),
        "num_images": c,
    }


gauss_counts = sum(stats[n]["count"] for n in GAUSSIAN_NOISES)
gauss_mse = sum(stats[n]["mse_sum"] for n in GAUSSIAN_NOISES)
gauss_psnr = sum(stats[n]["psnr_sum"] for n in GAUSSIAN_NOISES)
gauss_ssim = sum(stats[n]["ssim_sum"] for n in GAUSSIAN_NOISES)
gauss_time = sum(stats[n]["time_sum"] for n in GAUSSIAN_NOISES)

gaussian_agg = {
    "avg_mse": avg_or_none(gauss_mse, gauss_counts),
    "avg_psnr": avg_or_none(gauss_psnr, gauss_counts),
    "avg_ssim": avg_or_none(gauss_ssim, gauss_counts),
    "avg_runtime_ms": avg_or_none(gauss_time, gauss_counts),
    "num_images": gauss_counts,
}


sp_counts = sum(stats[n]["count"] for n in SALT_PEPPER_NOISES)
sp_mse = sum(stats[n]["mse_sum"] for n in SALT_PEPPER_NOISES)
sp_psnr = sum(stats[n]["psnr_sum"] for n in SALT_PEPPER_NOISES)
sp_ssim = sum(stats[n]["ssim_sum"] for n in SALT_PEPPER_NOISES)
sp_time = sum(stats[n]["time_sum"] for n in SALT_PEPPER_NOISES)

saltpepper_agg = {
    "avg_mse": avg_or_none(sp_mse, sp_counts),
    "avg_psnr": avg_or_none(sp_psnr, sp_counts),
    "avg_ssim": avg_or_none(sp_ssim, sp_counts),
    "avg_runtime_ms": avg_or_none(sp_time, sp_counts),
    "num_images": sp_counts,
}


total_count = gauss_counts + sp_counts
total_mse = gauss_mse + sp_mse
total_psnr = gauss_psnr + sp_psnr
total_ssim = gauss_ssim + sp_ssim
total_time = gauss_time + sp_time

overall_agg = {
    "avg_mse": avg_or_none(total_mse, total_count),
    "avg_psnr": avg_or_none(total_psnr, total_count),
    "avg_ssim": avg_or_none(total_ssim, total_count),
    "avg_runtime_ms": avg_or_none(total_time, total_count),
    "num_images": total_count,
}

final_results = {
    "per_noise": results_per_noise,
    "gaussian": gaussian_agg,
    "salt_pepper": saltpepper_agg,
    "overall": overall_agg,
}

with open(OUT_JSON, "w") as f:
    json.dump(final_results, f, indent=4)

print("\n=== DnCNN BSD68 BENCHMARK COMPLETE ===")
print("Saved JSON to:", OUT_JSON)
print(json.dumps(final_results, indent=4))
