import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageFrequencyProcessor:
    INP = "Proj3.tif"
    BPF_D0 = 2.0
    BPF_N  = 2.5
    LPF_D0 = 5.0
    LPF_N  = 1
    TARGET_PAIRS = 3
    RING_BAND_FRAC = 0.8
    NMS_WIN = 18
    DC_SUPPRESS_R = 11

    @staticmethod
    def fft2c(img):
        return np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))

    @staticmethod
    def ifft2c(F):
        return np.real(np.fft.ifft2(np.fft.ifftshift(F)))

    @staticmethod
    def normalize01(a):
        a = a.astype(np.float32)
        a -= a.min()
        a /= (a.max() + 1e-8)
        return a

    @staticmethod
    def logmag(F):
        return np.log1p(np.abs(F)).astype(np.float32)

    @staticmethod
    def radial(shape):
        h, w = shape
        Y, X = np.ogrid[:h, :w]
        cy, cx = h//2, w//2
        D = np.sqrt((Y - cy)**2 + (X - cx)**2).astype(np.float32)
        return D, cy, cx

    @staticmethod
    def butterworth_lowpass(D, D0, n):
        return 1.0 / (1.0 + (D / (D0 + 1e-8))**(2*n))

    @staticmethod
    def lobe_at(shape, center, D0, n):
        h, w = shape
        Y, X = np.ogrid[:h, :w]
        y0, x0 = center
        Dp = np.sqrt((Y - y0)**2 + (X - x0)**2).astype(np.float32)
        return 1.0 / (1.0 + (Dp / (D0 + 1e-8))**(2*n))

    @classmethod
    def build_sum_lobes(cls, shape, centers, D0, n, dc_stop_r=0):
        H = np.zeros(shape, np.float32)
        for (yy, xx) in centers:
            H = np.maximum(H, cls.lobe_at(shape, (yy, xx), D0, n))
        if dc_stop_r > 0:
            D, _, _ = cls.radial(shape)
            H *= (D > dc_stop_r).astype(np.float32)
        return np.clip(H, 0.0, 1.0)

    @staticmethod
    def refine_subpixel(M, y, x, win=7):
        r = win//2
        h, w = M.shape
        y0, y1 = max(0, y-r), min(h, y+r+1)
        x0, x1 = max(0, x-r), min(w, x+r+1)
        P = M[y0:y1, x0:x1]
        if P.size == 0 or P.max() <= 0:
            return float(y), float(x)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        W = P.astype(np.float32); s = W.sum()
        if s <= 0:
            return float(y), float(x)
        return float((W*yy).sum()/s), float((W*xx).sum()/s)

    @classmethod
    def estimate_ring_radius(cls, M, D):
        r = D.astype(np.int32)
        maxr = r.max()
        sums = np.bincount(r.ravel(), weights=M.ravel(), minlength=maxr+1)
        cnts = np.bincount(r.ravel(), minlength=maxr+1) + 1e-8
        prof = sums / cnts
        prof[:10] = prof.min()
        return int(np.argmax(prof))

    @classmethod
    def detect_peaks_first_ring(cls, Flog, target_pairs, band_frac, nms_win, dc_suppress_r):
        h, w = Flog.shape
        D, cy, cx = cls.radial(Flog.shape)
        M = Flog.copy()
        M[D < dc_suppress_r] = 0
        r0 = cls.estimate_ring_radius(M, D)
        band = np.abs(D - r0) <= (band_frac * r0)
        Ms = cv2.GaussianBlur(M, (5,5), 0); Ms[~band] = 0
        k = np.ones((nms_win, nms_win), np.uint8)
        Md = cv2.dilate(Ms, k)
        cand = (Ms == Md) & (Ms > 0)
        ys, xs = np.where(cand)
        scores = Ms[cand]
        if scores.size == 0:
            return []
        keep = xs >= cx
        ys, xs, scores = ys[keep], xs[keep], scores[keep]
        if scores.size == 0:
            return []
        order = np.argsort(-scores)
        ys, xs = ys[order][:target_pairs], xs[order][:target_pairs]
        peaks = []
        for y, x in zip(ys, xs):
            ry, rx = cls.refine_subpixel(Ms, int(y), int(x), win=7)
            peaks.append((ry, rx))
            peaks.append((h - 1 - ry, w - 1 - rx))
        return peaks

    @classmethod
    def run(cls):
        img = cv2.imread(cls.INP, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit("input image not found")
        F = cls.fft2c(img)
        Flog = cls.logmag(F)
        D, _, _ = cls.radial(img.shape)
        tries = [
            (cls.RING_BAND_FRAC, cls.NMS_WIN, cls.DC_SUPPRESS_R),
            (0.12, max(9, cls.NMS_WIN-2), max(2, cls.DC_SUPPRESS_R-2)),
            (0.15, max(7, cls.NMS_WIN-4), max(2, cls.DC_SUPPRESS_R-4)),
        ]
        peaks = []
        for bf, nms, dcs in tries:
            peaks = cls.detect_peaks_first_ring(Flog, cls.TARGET_PAIRS, bf, nms, dcs)
            if len(peaks) >= 2*max(3, cls.TARGET_PAIRS-2):
                break
        if not peaks:
            raise SystemExit("no peaks detected")
        H_bp = cls.build_sum_lobes(F.shape, peaks, cls.BPF_D0, cls.BPF_N, dc_stop_r=10)
        pat = cls.ifft2c(F * H_bp)
        pat_n = cls.normalize01(pat)
        H_lp = cls.butterworth_lowpass(D, cls.LPF_D0, cls.LPF_N)
        illum = cls.ifft2c(F * H_lp)
        uni = img.astype(np.float32) - illum
        uni += illum.mean()
        uni_n = cls.normalize01(uni)
        fig = plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(1,3,1); ax1.set_title("Frequency Filter Used"); ax1.imshow(H_bp, cmap="gray"); ax1.axis("off")
        ax2 = plt.subplot(1,3,2); ax2.set_title("Extracted Periodic Pattern"); ax2.imshow(pat_n, cmap="gray"); ax2.axis("off")
        ax3 = plt.subplot(1,3,3); ax3.set_title("Uniformly Illuminated Image"); ax3.imshow((uni_n*255).astype(np.uint8), cmap="gray"); ax3.axis("off")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    ImageFrequencyProcessor.run()
