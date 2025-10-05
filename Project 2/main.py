import numpy as np
import cv2
import matplotlib.pyplot as plt

class CardAlignCrop:
    def __init__(self):
        pass

    def to_gray(self, img):
        f = img.astype(np.float32)
        g = 0.114*f[:,:,0] + 0.587*f[:,:,1] + 0.299*f[:,:,2]
        return np.clip(g, 0, 255).astype(np.uint8)

    def gaussian_kernel(self, ksize=15, sigma=7.0):
        ax = np.arange(-(ksize//2), ksize//2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        k = np.exp(-(xx**2 + yy**2)/(2*sigma**2)).astype(np.float32)
        k /= k.sum()
        return k

    def conv2(self, x, k):
        return cv2.filter2D(x.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)

    def grad_xy(self, grayf):
        kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32)/3.0
        ky = kx.T
        Ix = self.conv2(grayf, kx)
        Iy = self.conv2(grayf, ky)
        return Ix, Iy

    def estimate_bg_gray(self, gray, rim=20):
        h, w = gray.shape
        rim = max(5, min(rim, h//4, w//4))
        strips = np.concatenate([gray[:rim,:].ravel(), gray[-rim:,:].ravel(),
                                 gray[:, :rim].ravel(), gray[:, -rim:].ravel()])
        return int(np.median(strips))

    def rotate_keep(self, img, angle_deg):
        bg = self.estimate_bg_gray(self.to_gray(img))
        h, w = img.shape[:2]; c = (w/2, h/2)
        M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
        cs, sn = abs(M[0,0]), abs(M[0,1])
        nw, nh = int(h*sn + w*cs), int(h*cs + w*sn)
        M[0,2] += (nw/2) - c[0]; M[1,2] += (nh/2) - c[1]
        return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(bg,bg,bg))

    def histogram_angle(self, gray):
        g_blur = self.conv2(gray, self.gaussian_kernel(15, 6.0))
        Ix, Iy = self.grad_xy(g_blur)
        mag = np.hypot(Ix, Iy)
        ang = (np.rad2deg(np.arctan2(Iy, Ix)) + 180.0) % 180.0
        m = mag > (0.25 * (mag.max() + 1e-6))
        if not np.any(m): return 0.0
        a = ang[m]
        a = np.where(a >= 90.0, 180.0 - a, a)
        w = mag[m]
        hist, edges = np.histogram(a, bins=90, range=(0.0, 90.0), weights=w)
        i = int(np.argmax(hist))
        return float(0.5*(edges[i]+edges[i+1]))

    def axis_alignment_score(self, img):
        g = self.to_gray(img)
        gf = self.conv2(g, self.gaussian_kernel(9, 3.0))
        Ix, Iy = self.grad_xy(gf)
        mag = np.hypot(Ix, Iy)
        m = mag > 0.25*(mag.max()+1e-6)
        if not np.any(m): return 0.0
        ang = (np.rad2deg(np.arctan2(Iy, Ix)) + 180.0) % 180.0
        a = np.where(ang[m] >= 90.0, 180.0 - ang[m], ang[m])
        w = mag[m]
        return float((w*((a<=0.5).astype(np.float32) + (a>=89.5).astype(np.float32))).sum())

    def choose_rotation(self, img, peak):
        candidates = [-peak, +peak, -(90.0 - peak), +(90.0 - peak)]
        best_angle, best_score = candidates[0], -1.0
        for a in candidates:
            s = self.axis_alignment_score(self.rotate_keep(img, a))
            if s > best_score:
                best_score, best_angle = s, a
        fine_range = np.arange(best_angle-2.0, best_angle+2.0+1e-9, 0.1)
        best_fine, best_s = best_angle, -1.0
        for a in fine_range:
            s = self.axis_alignment_score(self.rotate_keep(img, a))
            if s > best_s:
                best_s, best_fine = s, a
        return best_fine

    def otsu_1d(self, v):
        v = v.astype(np.float32)
        v = v - v.min()
        vmax = max(1e-6, v.max())
        u8 = np.clip(255.0 * (v / vmax), 0, 255).astype(np.uint8)
        hist = np.bincount(u8, minlength=256).astype(np.float64)
        p = hist / max(1, u8.size)
        w = np.cumsum(p); m = np.cumsum(p * np.arange(256)); mt = m[-1]
        denom = w*(1.0-w)
        with np.errstate(divide='ignore', invalid='ignore'):
            sb2 = ((mt*w - m)**2) / np.where(denom==0, np.nan, denom)
        t = int(np.nanargmax(sb2))
        return (u8 >= t)

    def crop_from_edges(self, edge_map, smooth_k=31, pad=2):
        rows = edge_map.sum(axis=1).astype(np.float32)
        cols = edge_map.sum(axis=0).astype(np.float32)
        k = smooth_k | 1
        ker = np.ones(k, np.float32)/k
        pad1 = k//2
        rs = np.convolve(np.pad(rows, (pad1,pad1), mode='edge'), ker, 'valid')
        cs = np.convolve(np.pad(cols, (pad1,pad1), mode='edge'), ker, 'valid')
        r_mask = self.otsu_1d(rs); c_mask = self.otsu_1d(cs)
        r_idx = np.where(r_mask)[0]; c_idx = np.where(c_mask)[0]
        if r_idx.size==0 or c_idx.size==0:
            return (0,0,edge_map.shape[1], edge_map.shape[0])
        y0, y1 = int(r_idx[0]), int(r_idx[-1])
        x0, x1 = int(c_idx[0]), int(c_idx[-1])
        y0 = max(0, y0-pad); x0 = max(0, x0-pad)
        y1 = min(edge_map.shape[0]-1, y1+pad); x1 = min(edge_map.shape[1]-1, x1+pad)
        return (x0, y0, x1-x0+1, y1-y0+1)

    def enforce_portrait(self, img):
        h, w = img.shape[:2]
        return self.rotate_keep(img, 90) if w > h else img

    def align_and_crop(self, path):
        img  = cv2.imread(path, cv2.IMREAD_COLOR)
        gray = self.to_gray(img)
        peak  = self.histogram_angle(gray)
        angle = self.choose_rotation(img, peak)
        rot   = self.rotate_keep(img, angle)

        gray_r = self.to_gray(rot)
        g_blur = self.conv2(gray_r, self.gaussian_kernel(15, 6.0))
        Ix, Iy = self.grad_xy(g_blur)
        mag = np.hypot(Ix, Iy)
        edge = (mag > 0.25*(mag.max()+1e-6)).astype(np.uint8) * 255
        x,y,w,h = self.crop_from_edges(edge, smooth_k=31, pad=2)
        crop = rot[y:y+h, x:x+w].copy()

        gray_c   = self.to_gray(crop)
        g_blur_c = self.conv2(gray_c, self.gaussian_kernel(15, 6.0))
        Ix_c, Iy_c = self.grad_xy(g_blur_c)
        mag_c = np.hypot(Ix_c, Iy_c)
        edge_c = (mag_c > 0.28*(mag_c.max()+1e-6)).astype(np.uint8) * 255
        x2,y2,w2,h2 = self.crop_from_edges(edge_c, smooth_k=27, pad=0)
        crop = crop[y2:y2+h2, x2:x2+w2].copy()

        crop = self.enforce_portrait(crop)

        plt.figure(figsize=(11,4))
        plt.subplot(1,2,1); plt.title("Figure 1. Input Image"); plt.axis('off'); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.subplot(1,2,2); plt.title("Figure 2. Output Image – aligned and cropped"); plt.axis('off'); plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.tight_layout(); plt.show()
        return crop

if __name__ == "__main__":
    path = input("Enter image path: ").strip()
    CardAlignCrop().align_and_crop(path)
