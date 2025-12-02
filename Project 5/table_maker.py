import os
import json
import pandas as pd
import numpy as np

METRIC_DIR = "assets/metrics"
OUTPUT_DIR = "tables_json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGES = ["lenna", "nature", "potrait", "street"]
METHODS = ["smoothing", "median", "adaptive", "dncnn", "restormer"]
NOISE_TYPES = ["gauss15", "gauss25", "gauss50", "sp2", "sp5"]


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_method(fname):
    base = fname.replace(".json", "")
    parts = base.split("_")

    if len(parts) == 1:
        return "classical"

    return parts[1]  # dncnn or restormer


def load_all():
    """Load classical + dncnn + restormer JSONs into a single dict per image."""
    all_data = {img: {} for img in IMAGES}

    for fname in os.listdir(METRIC_DIR):
        if not fname.endswith(".json"):
            continue

        method = extract_method(fname)

        for img in IMAGES:
            if fname.startswith(img):
                all_data[img][method] = load_json(os.path.join(METRIC_DIR, fname))

    return all_data


def format_cell(m):
    if m is None:
        return None
    return {
        "mse": m["mse"],
        "psnr": m["psnr"],
        "ssim": m["ssim"],
        "runtime": m["runtime_ms"]
    }


def extract_metrics(img_data, noise, method):
    """Correct extractor for classical + dncnn + restormer files."""
    if method in ["smoothing", "median", "adaptive"]:
        return img_data.get("classical", {}).get(noise, {}).get(method)

    if method == "dncnn":
        return img_data.get("dncnn", {}).get(noise, {}).get("dncnn")

    if method == "restormer":
        block = img_data.get("restormer", {})
        if noise in block:
            # Restormer may store as "restormer" or "restormer_sigma50"
            if "restormer" in block[noise]:
                return block[noise]["restormer"]
            if "restormer_sigma50" in block[noise]:
                return block[noise]["restormer_sigma50"]

    return None


# ----------------------------------------------------------
# Build per-image table
# ----------------------------------------------------------
def build_table_json(img_name, img_data):
    tbl = {}

    for noise in NOISE_TYPES:
        tbl[noise] = {}
        for method in METHODS:
            m = extract_metrics(img_data, noise, method)
            tbl[noise][method] = format_cell(m)

    return tbl


# ----------------------------------------------------------
# Collect global metrics
# ----------------------------------------------------------
def extract_all_metrics(all_data):
    records = []

    for img in IMAGES:
        img_data = all_data[img]

        for noise in NOISE_TYPES:
            for method in METHODS:

                m = extract_metrics(img_data, noise, method)
                if m is None:
                    continue

                records.append({
                    "image": img,
                    "noise": noise,
                    "method": method,
                    "mse": m["mse"],
                    "psnr": m["psnr"],
                    "ssim": m["ssim"],
                    "runtime": m["runtime_ms"]
                })

    return pd.DataFrame(records)


def normalize(series, invert=False):
    if series.max() == series.min():
        return np.ones_like(series)
    if invert:
        return 1 - (series - series.min()) / (series.max() - series.min())
    return (series - series.min()) / (series.max() - series.min())


# ----------------------------------------------------------
# Gaussian vs SP Leaderboard
# ----------------------------------------------------------
def build_noise_specific_scores(df):
    df_norm = df.copy()

    df_norm["mse_n"] = normalize(df["mse"], invert=True)
    df_norm["psnr_n"] = normalize(df["psnr"])
    df_norm["ssim_n"] = normalize(df["ssim"])
    df_norm["runtime_n"] = normalize(df["runtime"], invert=True)

    df_norm["score"] = (
        df_norm["mse_n"]
        + df_norm["psnr_n"]
        + df_norm["ssim_n"]
        + df_norm["runtime_n"]
    ) / 4.0

    gaussian = df_norm[df_norm.noise.isin(["gauss15", "gauss25", "gauss50"])]
    sp = df_norm[df_norm.noise.isin(["sp2", "sp5"])]

    leaderboard = {
        "gaussian": gaussian.groupby("method")["score"].mean().sort_values(ascending=False).to_dict(),
        "salt_pepper": sp.groupby("method")["score"].mean().sort_values(ascending=False).to_dict(),
        "overall": df_norm.groupby("method")["score"].mean().sort_values(ascending=False).to_dict()
    }

    return leaderboard


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    all_data = load_all()

    # 1. Per-image JSON tables
    for img in IMAGES:
        tbl = build_table_json(img, all_data[img])
        out_path = os.path.join(OUTPUT_DIR, f"{img}_table.json")
        json.dump(tbl, open(out_path, "w"), indent=4)
        print("Saved:", out_path)

    # 2. Global metrics dataframe
    df_all = extract_all_metrics(all_data)

    # 3. Gaussian vs SP leaderboard
    leaderboard = build_noise_specific_scores(df_all)
    json.dump(leaderboard, open(os.path.join(OUTPUT_DIR, "global_leaderboard.json"), "w"), indent=4)

    print("\nSaved: global_leaderboard.json\n")
