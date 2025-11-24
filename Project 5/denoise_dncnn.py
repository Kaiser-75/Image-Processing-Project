import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from metrics import mse_np, psnr_np, ssim_np

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
BASE = "assets"
NOISE_DIR = os.path.join(BASE, "noise")
OUT_DIR = os.path.join(BASE, "output_dncnn")
MODEL_PATH = "assets/weights/dncnn_gray_blind.pth"
JSON_DIR = os.path.join(BASE, "metrics")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"



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


print("Loading:", MODEL_PATH)

state = torch.load(MODEL_PATH, map_location="cpu")

# Fix missing prefix
if not list(state.keys())[0].startswith("model."):
    state = {"model." + k: v for k, v in state.items()}

model = DnCNN(num_layers=20).to(device)
model.load_state_dict(state, strict=False)
model.eval()


def load_clean(base_name):
    png = os.path.join(BASE, base_name + ".png")
    jpg = os.path.join(BASE, base_name + ".jpg")

    if os.path.exists(png):
        return np.array(Image.open(png).convert("L")).astype(np.float32)
    elif os.path.exists(jpg):
        return np.array(Image.open(jpg).convert("L")).astype(np.float32)
    else:
        return None

def dncnn_denoise(arr):
    x = arr.astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model(x).cpu().squeeze().numpy()
    t1 = time.time()

    den = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    runtime = (t1 - t0) * 1000
    return den, runtime



for fname in os.listdir(NOISE_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    noisy_path = os.path.join(NOISE_DIR, fname)
    noisy_np = np.array(Image.open(noisy_path).convert("L")).astype(np.float32)

    base_name = fname.split("_")[0]         
    noise_label = fname.split("_")[1].split(".")[0]  

    clean_np = load_clean(base_name)
    if clean_np is None:
        print("Clean image missing for", fname)
        continue

    json_path = os.path.join(JSON_DIR, f"{base_name}_dncnn.json")


    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}


    den, runtime = dncnn_denoise(noisy_np)

  
    out_path = os.path.join(OUT_DIR, f"dncnn_{fname}")
    Image.fromarray(den).save(out_path)


    mse = mse_np(clean_np, den)
    psnr = psnr_np(clean_np, den)
    ssim = ssim_np(clean_np, den)

   
    if noise_label not in data:
        data[noise_label] = {}

    data[noise_label]["dncnn"] = {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "runtime_ms": round(runtime, 3)
    }


    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print("Updated JSON for", fname)

print("DnCNN metrics complete.")
