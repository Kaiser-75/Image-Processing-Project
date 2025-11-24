import os
import numpy as np
from PIL import Image

# Path
BASE = r"C:\Users\kaise\Documents\Project 5\assets"
NOISE_DIR = os.path.join(BASE, "noise")
os.makedirs(NOISE_DIR, exist_ok=True)

# Noise settings
gaussian_sigmas = [15, 25, 50]
sp_amounts = [0.02, 0.05]

# Gaussian noise
def add_gaussian_noise(img_np, sigma):
    noise = np.random.normal(0, sigma, img_np.shape)
    noisy = img_np + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

# Salt & Pepper noise
def add_salt_pepper_noise(img_np, amount):
    noisy = img_np.copy()
    num_pixels = img_np.size
    num_sp = int(num_pixels * amount)

    # Salt (white)
    coords = np.unravel_index(np.random.choice(num_pixels, num_sp // 2, replace=False), img_np.shape)
    noisy[coords] = 255

    # Pepper (black)
    coords = np.unravel_index(np.random.choice(num_pixels, num_sp // 2, replace=False), img_np.shape)
    noisy[coords] = 0

    return noisy.astype(np.uint8)


for fname in os.listdir(BASE):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):

        in_path = os.path.join(BASE, fname)
        img = Image.open(in_path).convert("L").resize((512, 512))
        img_np = np.array(img).astype(np.float32)

        name = os.path.splitext(fname)[0]

        # Gaussian noise outputs
        for sigma in gaussian_sigmas:
            noisy = add_gaussian_noise(img_np, sigma)
            out_path = os.path.join(NOISE_DIR, f"{name}_gauss{sigma}.png")
            Image.fromarray(noisy).save(out_path)

        # Salt & Pepper outputs
        for amount in sp_amounts:
            noisy = add_salt_pepper_noise(img_np, amount)
            out_path = os.path.join(NOISE_DIR, f"{name}_sp{int(amount*100)}.png")
            Image.fromarray(noisy).save(out_path)

print("Noise generation completed.")
