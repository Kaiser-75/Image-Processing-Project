import os
import time
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from basicsr.models.archs.restormer_arch import Restormer
from metrics import mse_np, psnr_np, ssim_np


CLEAN_DIR = "assets/gray_bsd"
NOISE_DIR = "assets/noise_bsd"
OUT_JSON_FULL = "assets/metrics/bsd68_restormer.json"
OUT_JSON_SUMMARY = "assets/metrics/bsd68_restormer_summary.json"

WEIGHT_PATH = "assets/weights/gaussian_gray_denoising_blind.pth"

os.makedirs(os.path.dirname(OUT_JSON_FULL), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)


print("Loading Restormer blind model...")

model = Restormer(
    inp_channels=1,
    out_channels=1,
    dim=48,
    num_blocks=[4, 6, 6, 8],
    num_refinement_blocks=4,
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type="BiasFree",
    dual_pixel_task=False,
)

ckpt = torch.load(WEIGHT_PATH, map_location="cpu")
state = ckpt["params"] if "params" in ckpt else ckpt
model.load_state_dict(state, strict=True)

model = model.to(device).eval()

print("Restormer loaded.")



def denoise_restormer(img_np):
    img = img_np.astype(np.float32) / 255.0
    img = img[None, None, :, :]

    x = torch.from_numpy(img).to(device)
    _, _, h, w = x.shape

    # pad to multiple of 8
    factor = 8
    H = int(math.ceil(h / factor) * factor)
    W = int(math.ceil(w / factor) * factor)

    padh = H - h
    padw = W - w

    x_pad = F.pad(x, (0, padw, 0, padh), mode="reflect")

    with torch.no_grad():
        out = model(x_pad)
        out = out[:, :, :h, :w]

    out = out.squeeze().cpu().clamp(0, 1).numpy()
    return (out * 255.0).astype(np.uint8)



noise_results = {}
gaussian_metrics = {"mse": [], "psnr": [], "ssim": [], "time": []}
sp_metrics = {"mse": [], "psnr": [], "ssim": [], "time": []}

GAUSS_TYPES = ["gauss15", "gauss25", "gauss50"]
SP_TYPES = ["sp2", "sp5"]



for fname in os.listdir(NOISE_DIR):

    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    name = fname.split("_")[0]
    noise_label = fname.replace(name + "_", "").split(".")[0]

    clean_path_png = os.path.join(CLEAN_DIR, name + ".png")
    clean_path_jpg = os.path.join(CLEAN_DIR, name + ".jpg")

    if os.path.exists(clean_path_png):
        clean_np = np.array(Image.open(clean_path_png).convert("L"))
    elif os.path.exists(clean_path_jpg):
        clean_np = np.array(Image.open(clean_path_jpg).convert("L"))
    else:
        print("Missing clean image:", name)
        continue

    noisy_np = np.array(Image.open(os.path.join(NOISE_DIR, fname)).convert("L"))

    # inference
    t0 = time.time()
    den = denoise_restormer(noisy_np)
    runtime = (time.time() - t0) * 1000.0

    mse = mse_np(clean_np, den)
    psnr = psnr_np(clean_np, den)
    ssim = ssim_np(clean_np, den)

  
    if noise_label not in noise_results:
        noise_results[noise_label] = {}

    noise_results[noise_label][name] = {
        "restormer": {
            "mse": float(mse),
            "psnr": float(psnr),
            "ssim": float(ssim),
            "runtime_ms": round(runtime, 3)
        }
    }

   
    if noise_label in GAUSS_TYPES:
        gaussian_metrics["mse"].append(mse)
        gaussian_metrics["psnr"].append(psnr)
        gaussian_metrics["ssim"].append(ssim)
        gaussian_metrics["time"].append(runtime)

    elif noise_label in SP_TYPES:
        sp_metrics["mse"].append(mse)
        sp_metrics["psnr"].append(psnr)
        sp_metrics["ssim"].append(ssim)
        sp_metrics["time"].append(runtime)

    print(f"[{noise_label}] {fname} -> PSNR={psnr:.2f}, SSIM={ssim:.4f}, Time={runtime:.2f}ms")



with open(OUT_JSON_FULL, "w") as f:
    json.dump(noise_results, f, indent=4)

print("\nSaved full JSON:", OUT_JSON_FULL)


def avg(x):
    return float(np.mean(x)) if len(x) else None

summary = {
    "gaussian": {
        "avg_mse": avg(gaussian_metrics["mse"]),
        "avg_psnr": avg(gaussian_metrics["psnr"]),
        "avg_ssim": avg(gaussian_metrics["ssim"]),
        "avg_runtime_ms": avg(gaussian_metrics["time"]),
        "count": len(gaussian_metrics["mse"]),
    },
    "salt_pepper": {
        "avg_mse": avg(sp_metrics["mse"]),
        "avg_psnr": avg(sp_metrics["psnr"]),
        "avg_ssim": avg(sp_metrics["ssim"]),
        "avg_runtime_ms": avg(sp_metrics["time"]),
        "count": len(sp_metrics["mse"]),
    }
}

with open(OUT_JSON_SUMMARY, "w") as f:
    json.dump(summary, f, indent=4)

print("Saved summary JSON:", OUT_JSON_SUMMARY)
print("\nBenchmark complete.")
