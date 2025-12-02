import os
import json
import numpy as np
import matplotlib.pyplot as plt

METRIC_DIR = "assets/metrics"
TABLES_DIR = "tables"
OUT_DIR = "plots"

os.makedirs(OUT_DIR, exist_ok=True)

IMAGES = ["lenna", "nature", "potrait", "street"]
METHODS = ["smoothing", "median", "adaptive", "dncnn", "restormer"]
NOISE_TYPES = ["gauss15", "gauss25", "gauss50", "sp2", "sp5"]

# ------------------------------------------------------------------
# Load a single per image table
# ------------------------------------------------------------------
def load_table(img_name):
    path = os.path.join(TABLES_DIR, f"{img_name}_table.json")
    with open(path, "r") as f:
        return json.load(f)

# ------------------------------------------------------------------
# Extract arrays for plotting
# ------------------------------------------------------------------
def extract_metric(table, metric_key):
    data = {}
    for method in METHODS:
        arr = []
        for noise in NOISE_TYPES:
            m = table[noise].get(method, None)
            if m is None:
                arr.append(np.nan)
            else:
                arr.append(m[metric_key])
        data[method] = np.array(arr)
    return data

# ------------------------------------------------------------------
# Plot line chart for PSNR, MSE, SSIM
# ------------------------------------------------------------------
def plot_line(metric_name, metric_key, img_name, table):
    metric_data = extract_metric(table, metric_key)

    plt.figure(figsize=(7,5))
    for method, vals in metric_data.items():
        plt.plot(NOISE_TYPES, vals, marker="o", linewidth=2, label=method)

    plt.xlabel("Noise type")
    plt.ylabel(metric_name)
    plt.title(f"{img_name} {metric_name}")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{img_name}_{metric_name}.png"))
    plt.close()

# ------------------------------------------------------------------
# Runtime bar plot
# ------------------------------------------------------------------
def plot_runtime(img_name, table):
    runtime_data = extract_metric(table, "runtime")

    plt.figure(figsize=(7,5))
    width = 0.15
    x = np.arange(len(NOISE_TYPES))

    for idx, method in enumerate(METHODS):
        plt.bar(x + idx*width, runtime_data[method], width, label=method)

    plt.xticks(x + width*2, NOISE_TYPES)
    plt.ylabel("Runtime ms")
    plt.title(f"{img_name} runtime")
    plt.grid(axis="y", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{img_name}_runtime.png"))
    plt.close()

# ------------------------------------------------------------------
# Gaussian vs SP global leaderboard
# ------------------------------------------------------------------
def load_leaderboard():
    with open(os.path.join(TABLES_DIR, "global_leaderboard.json"), "r") as f:
        return json.load(f)

def plot_leaderboard():
    lb = load_leaderboard()

    gaussian = lb["gaussian"]
    sp = lb["salt_pepper"]
    overall = lb["overall"]

    groups = ["gaussian", "salt_pepper", "overall"]
    methods = list(overall.keys())

    plt.figure(figsize=(8,5))

    x = np.arange(len(methods))
    width = 0.25

    g_vals = [gaussian[m] for m in methods]
    s_vals = [sp[m] for m in methods]
    o_vals = [overall[m] for m in methods]

    plt.bar(x - width, g_vals, width, label="gaussian")
    plt.bar(x, s_vals, width, label="salt pepper")
    plt.bar(x + width, o_vals, width, label="overall")

    plt.xticks(x, methods)
    plt.ylabel("Score")
    plt.title("Noise leaderboard")
    plt.grid(axis="y", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "leaderboard.png"))
    plt.close()

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
if __name__ == "__main__":

    for img in IMAGES:
        table = load_table(img)

        plot_line("PSNR", "psnr", img, table)
        plot_line("MSE", "mse", img, table)
        plot_line("SSIM", "ssim", img, table)
        plot_runtime(img, table)

    plot_leaderboard()

    print("Plots saved in plots/")
