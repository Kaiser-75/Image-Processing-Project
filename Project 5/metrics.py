import numpy as np

def mse_np(img1, img2):
    err = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    return float(err)


def psnr_np(img1, img2, data_range=255.0):
    mse = mse_np(img1, img2)
    if mse == 0:
        return float("inf")
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return float(psnr)


def ssim_np(img1, img2, data_range=255.0):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    K1 = 0.01
    K2 = 0.03
    L = data_range

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)

    return float(num / den)
