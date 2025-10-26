import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class MatchResult:
    ok: bool
    rank: str = "Unknown"
    suit: str = "Unknown"
    rank_score: float = -1.0
    suit_score: float = -1.0

class Template:
    def __init__(self, name: str, img: np.ndarray):
        self.name = name
        self.img = img

class CardMatcher:
    def __init__(self, root: str = "assets/templates"):
        self.root = Path(root)
        self.rank_dir = self.root / "ranks"
        self.suit_dir = self.root / "suits"
        self.ranks: List[Template] = []
        self.suits: List[Template] = []
        self._load_templates()

    def _bin(self, img: np.ndarray) -> np.ndarray:
        _, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out

    def _load_dir(self, path: Path) -> List[Template]:
        arr = []
        for p in sorted(path.glob("*.png")):
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            arr.append(Template(p.stem, self._bin(im)))
        return arr

    def _load_templates(self) -> None:
        self.ranks = self._load_dir(self.rank_dir)
        self.suits = self._load_dir(self.suit_dir)

    def _score(self, query: np.ndarray, tmpl: np.ndarray) -> float:
        if query.shape != tmpl.shape:
            return -1.0
        ncc = cv2.matchTemplate(query, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0]
        inv = 255 - tmpl
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)
        q = (query > 0).astype(np.float32)
        dsc = float((q * dist).mean())
        return 0.5 * ncc + 0.5 * dsc

    def _best_match(self, query: np.ndarray, templates: List[Template]) -> Tuple[str, float]:
        best_name, best_score = "Unknown", -1.0
        for t in templates:
            score = self._score(query, t.img)
            if score > best_score:
                best_name, best_score = t.name, score
        return best_name, best_score

    def match(self, rank_img: Optional[np.ndarray], suit_img: Optional[np.ndarray]) -> MatchResult:
        if rank_img is None or suit_img is None:
            return MatchResult(False)
        q1 = self._bin(rank_img)
        q2 = self._bin(suit_img)
        r_name, r_score = self._best_match(q1, self.ranks)
        s_name, s_score = self._best_match(q2, self.suits)
        ok = r_name != "Unknown" and s_name != "Unknown"
        return MatchResult(ok, r_name, s_name, r_score, s_score)

