import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

CORNER_WIDTH = 32
CORNER_HEIGHT = 92
RANK_SIZE = (70, 125)
SUIT_SIZE = (70, 100)

@dataclass
class CardCrops:
    ok: bool
    reason: str = ""
    warp_gray: Optional[np.ndarray] = None
    quad: Optional[np.ndarray] = None
    rank_img: Optional[np.ndarray] = None
    suit_img: Optional[np.ndarray] = None

class CardDetector:
    def __init__(self):
        self.corner_w = CORNER_WIDTH
        self.corner_h = CORNER_HEIGHT
        self.rank_size = RANK_SIZE
        self.suit_size = SUIT_SIZE
        self.maxW, self.maxH = 200, 300

    def _find_contours(self, img_bin):
        res = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(res) == 2:
            cnts, _ = res
        else:
            _, cnts, _ = res
        return cnts

    def _preprocess_image(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _order_four(self, pts):
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], np.float32)

    def _quad_from_points(self, points):
        p = np.squeeze(points, axis=1).astype(np.float32)
        if p.shape[0] != 4:
            rect = cv2.minAreaRect(p)
            box = cv2.boxPoints(rect).astype(np.float32)
            return self._order_four(box)
        return self._order_four(p)

    def _warp_card(self, image_bgr, approx):
        quad = self._quad_from_points(approx)
        dst = np.array([[0, 0], [self.maxW-1, 0], [self.maxW-1, self.maxH-1], [0, self.maxH-1]], np.float32)
        M = cv2.getPerspectiveTransform(quad, dst)
        warp = cv2.warpPerspective(image_bgr, M, (self.maxW, self.maxH))
        return warp, quad

    def _choose_best_rotation(self, warp_bgr):
        best_rot, best_score = 0, -1
        best_gray = None
        cur = warp_bgr.copy()
        for k in range(4):
            gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
            corner = gray[0:self.corner_h, 0:self.corner_w]
            corner = cv2.resize(corner, (0, 0), fx=4, fy=4)
            _, binv = cv2.threshold(corner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            black = np.sum(binv > 0)
            if black > best_score:
                best_score = black
                best_gray = gray
                best_rot = k
            cur = cv2.rotate(cur, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return best_gray, best_rot

    def _extract_rank_suit(self, corner_bin):
        rank_img = suit_img = None
        h = corner_bin.shape[0]
        half = h // 2
        rank_half = corner_bin[0:half, :]
        suit_half = corner_bin[half:, :]
        cnts_r = self._find_contours(rank_half)
        if cnts_r:
            c = max(cnts_r, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            roi = rank_half[y:y+h, x:x+w]
            rank_img = cv2.resize(roi, self.rank_size, interpolation=cv2.INTER_AREA)
        cnts_s = self._find_contours(suit_half)
        if cnts_s:
            c = max(cnts_s, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            roi = suit_half[y:y+h, x:x+w]
            suit_img = cv2.resize(roi, self.suit_size, interpolation=cv2.INTER_AREA)
        return rank_img, suit_img

    def _find_cards(self, thresh_image):
        cnts = self._find_contours(thresh_image)
        if not cnts:
            return []
        H, W = thresh_image.shape[:2]
        area_min, area_max = 0.02 * H * W, 0.6 * H * W
        valid = []
        for c in cnts:
            a = cv2.contourArea(c)
            if area_min < a < area_max:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) >= 4:
                    valid.append(c)
        return valid

    def _preprocess_card(self, contour, image_bgr):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        warp_bgr, quad_pts = self._warp_card(image_bgr, approx)
        warp_gray, rot = self._choose_best_rotation(warp_bgr)
        for _ in range(rot):
            warp_gray = cv2.rotate(warp_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        corner = warp_gray[0:self.corner_h, 0:self.corner_w]
        corner = cv2.resize(corner, (0, 0), fx=4, fy=4)
        _, binv = cv2.threshold(corner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        rank_img, suit_img = self._extract_rank_suit(binv)
        return warp_gray, quad_pts, rank_img, suit_img

    def from_frame(self, frame_bgr):
        thresh = self._preprocess_image(frame_bgr)
        cards = self._find_cards(thresh)
        if not cards:
            return CardCrops(False, "no_contours")
        for c in cards:
            warp, quad, rank_img, suit_img = self._preprocess_card(c, frame_bgr)
            if rank_img is not None and suit_img is not None:
                return CardCrops(True, "", warp, quad, rank_img, suit_img)
        return CardCrops(False, "no_card")
