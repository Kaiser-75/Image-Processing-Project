import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from basicsr.models.archs.restormer_arch import Restormer



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
    return median_filter(img_np, size=3).astype(np.uint8)


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
    W = local_var / (local_var + noise_var + 1e-8)
    filtered = local_mean + W * (img - local_mean)

    return np.clip(filtered, 0, 255).astype(np.uint8)



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



def load_dncnn(weight_path, device):
    print("Loading DnCNN...")
    model = DnCNN().to(device)
    state = torch.load(weight_path, map_location="cpu")

    if not list(state.keys())[0].startswith("model."):
        state = {f"model.{k}": v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_restormer(weight_path, device):
    print("Loading Restormer...")
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
    ckpt = torch.load(weight_path, map_location="cpu")
    state = ckpt["params"] if "params" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model



def dncnn_denoise(model, img_np, device):
    x = img_np.astype(np.float32)/255.0
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model(x).cpu().squeeze().numpy()
    t1 = time.time()

    den = np.clip(out*255.0, 0, 255).astype(np.uint8)
    return den, (t1 - t0)*1000


def restormer_denoise(model, img_np, device):
    x = img_np.astype(np.float32)/255.0
    x = torch.from_numpy(x[None,None,:,:]).to(device)

    _,_,h,w = x.shape
    H = int(math.ceil(h/8)*8)
    W = int(math.ceil(w/8)*8)

    x_pad = F.pad(x,(0,W-w,0,H-h), mode="reflect")

    t0 = time.time()
    with torch.no_grad():
        out = model(x_pad)
        out = out[:,:,:h,:w]
    t1 = time.time()

    out = out.squeeze().cpu().clamp(0,1).numpy()
    return (out*255).astype(np.uint8), (t1 - t0)*1000



def main():

    print("\n=== Image Denoising ===")
    noisy_path = input("Enter path to noisy image: ").strip()

    if not os.path.exists(noisy_path):
        print("File not found. Exiting.")
        return

    noisy = np.array(Image.open(noisy_path).convert("L"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load deep models
    dncnn = load_dncnn("assets/weights/dncnn_gray_blind.pth", device)
    restormer = load_restormer("assets/weights/gaussian_gray_denoising_blind.pth", device)

    # classical methods
    t0 = time.time(); sm = smoothing_filter(noisy); t_sm = (time.time()-t0)*1000
    t0 = time.time(); md = median_filter_custom(noisy); t_md = (time.time()-t0)*1000
    t0 = time.time(); ad = lee_filter(noisy); t_ad = (time.time()-t0)*1000

    # deep models
    dncnn_out, t_dn = dncnn_denoise(dncnn, noisy, device)
    rest_out, t_rs = restormer_denoise(restormer, noisy, device)

    # show comparison
    titles = [
        "Noisy Input",
        f"Smoothing\n({t_sm:.1f} ms)",
        f"Median\n({t_md:.1f} ms)",
        f"Adaptive Lee\n({t_ad:.1f} ms)",
        f"DnCNN\n({t_dn:.1f} ms)",
        f"Restormer\n({t_rs:.1f} ms)"
    ]
    images = [noisy, sm, md, ad, dncnn_out, rest_out]

    plt.figure(figsize=(18,5))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(1,6,i+1)
        plt.imshow(img, cmap="gray")
        plt.title(title, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
