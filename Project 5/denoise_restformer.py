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



BASE = "assets"
NOISE_DIR = os.path.join(BASE, "noise")
OUT_IMG = os.path.join(BASE, "output_restormer")
OUT_JSON = os.path.join(BASE, "metrics")

WEIGHT_PATH = os.path.join(BASE, "weights", "gaussian_gray_denoising_blind.pth")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_JSON, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading Restormer BLIND grayscale model...")

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

model = model.to(device)
model.eval()

print("Restormer BLIND loaded.")



def denoise_restormer_blind(img_np: np.ndarray) -> np.ndarray:
    """
    img_np: H x W, uint8 grayscale
    returns: H x W, uint8 grayscale
    """
    img = img_np.astype(np.float32) / 255.0        # [0,1]
    img = img[None, None, :, :]                    # [1,1,H,W]

    x = torch.from_numpy(img).to(device)

    _, _, h, w = x.shape
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



for fname in os.listdir(NOISE_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    noisy_path = os.path.join(NOISE_DIR, fname)
    noisy_np = np.array(Image.open(noisy_path).convert("L"))

    base = fname.split("_")[0]
    clean_png = os.path.join(BASE, base + ".png")
    clean_jpg = os.path.join(BASE, base + ".jpg")

    if os.path.exists(clean_png):
        clean_np = np.array(Image.open(clean_png).convert("L"))
    elif os.path.exists(clean_jpg):
        clean_np = np.array(Image.open(clean_jpg).convert("L"))
    else:
        print("Missing clean image for", fname)
        continue

    t0 = time.time()
    out = denoise_restormer_blind(noisy_np)
    runtime = (time.time() - t0) * 1000.0

    out_path = os.path.join(OUT_IMG, "restormer_blind_" + fname)
    Image.fromarray(out).save(out_path)


    mse = mse_np(clean_np, out)
    psnr = psnr_np(clean_np, out)
    ssim = ssim_np(clean_np, out)


    noise_key = (
        fname.replace(base + "_", "")
        .replace(".png", "")
        .replace(".jpg", "")
        .replace(".jpeg", "")
    )

    json_name = os.path.join(OUT_JSON, base + "_restormer.json")
    if os.path.exists(json_name):
        with open(json_name, "r") as f:
            data = json.load(f)
    else:
        data = {}

    if noise_key not in data:
        data[noise_key] = {}

    data[noise_key]["restormer"] = {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "runtime_ms": round(runtime, 3),
    }

    with open(json_name, "w") as f:
        json.dump(data, f, indent=4)

    print("Updated", json_name)

print("Restormer BLIND denoising complete.")
