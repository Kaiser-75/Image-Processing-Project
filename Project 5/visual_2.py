import os
import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
TABLES_DIR = "tables"
OUT_DIR = "plots"

os.makedirs(OUT_DIR, exist_ok=True)

IMAGES = ["lenna", "nature", "potrait", "street"]
METHODS = ["smoothing", "median", "adaptive", "dncnn", "restormer"]
NOISE_TYPES = ["gauss15", "gauss25", "gauss50", "sp2", "sp5"]


# -------------------------------------------------------------------------
# LOAD TABLE
# -------------------------------------------------------------------------
def load_table(img_name):
    path = os.path.join(TABLES_DIR, f"{img_name}_table.json")
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------------------------------------
# EXTRACT METRIC
# -------------------------------------------------------------------------
def extract_metric(table, metric_key):
    data = {}
    for method in METHODS:
        vals = []
        for noise in NOISE_TYPES:
            entry = table[noise].get(method)
            vals.append(entry[metric_key] if entry else np.nan)
        data[method] = np.array(vals)
    return data


# -------------------------------------------------------------------------
# RADAR PLOT (NORMALIZED + MARKED RADIAL TICKS)
# -------------------------------------------------------------------------
def plot_radar(metric_name, metric_key, metric_data, filename, title=None):
    labels = NOISE_TYPES
    num_vars = len(labels)

    # Collect all values for global normalization
    all_vals = np.concatenate([metric_data[m] for m in METHODS])
    min_v, max_v = np.nanmin(all_vals), np.nanmax(all_vals)

    # Normalize
    norm_data = {m: (metric_data[m] - min_v) / (max_v - min_v + 1e-8) for m in METHODS}

    # Angles
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Figure
    plt.figure(figsize=(9, 9), facecolor="white")
    ax = plt.subplot(111, polar=True, facecolor="white")

    # Plot each method
    for method in METHODS:
        vals = norm_data[method].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.4, marker="o", label=method)
        ax.fill(angles, vals, alpha=0.10)

    # NO base xticks
    ax.set_xticks([])
    # Manually draw labels outward
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, 1.20, label, fontsize=16, fontweight="bold",
                ha="center", va="center")

    # Radial ticks (show normalized scale)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"], fontsize=13)
    ax.set_rlabel_position(0)

    # Grid
    ax.grid(color="gray", linestyle=":", linewidth=1.2)

    # Optional title
    if title:
        plt.title(title, fontsize=20, pad=25, fontweight="bold")

    # Legend
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close()


# -------------------------------------------------------------------------
# AVERAGE ACROSS ALL IMAGES
# -------------------------------------------------------------------------
def compute_average(metric_key):
    all_tables = [load_table(img) for img in IMAGES]

    avg_data = {m: [] for m in METHODS}

    for noise in NOISE_TYPES:
        for method in METHODS:
            vals = []
            for tbl in all_tables:
                row = tbl[noise].get(method)
                if row:
                    vals.append(row[metric_key])
            avg_data[method].append(np.mean(vals))

    # Convert to arrays
    return {m: np.array(avg_data[m]) for m in METHODS}


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
if __name__ == "__main__":

    # -----------------------------
    # INDIVIDUAL IMAGE PLOTS
    # -----------------------------
    for img in IMAGES:
        table = load_table(img)

        psnr_data = extract_metric(table, "psnr")
        plot_radar("PSNR", "psnr", psnr_data,
                   filename=f"{img}_PSNR_radar.png",
                   title=None)

        ssim_data = extract_metric(table, "ssim")
        plot_radar("SSIM", "ssim", ssim_data,
                   filename=f"{img}_SSIM_radar.png",
                   title=None)

    # -----------------------------
    # GLOBAL AVERAGE PLOTS
    # -----------------------------
    avg_psnr = compute_average("psnr")
    plot_radar("PSNR", "psnr", avg_psnr,
               filename="AVERAGE_PSNR_radar.png"
              )

    avg_ssim = compute_average("ssim")
    plot_radar("SSIM", "ssim", avg_ssim,
               filename="AVERAGE_SSIM_radar.png"
              )

    print("All radar plots saved in plots/")
