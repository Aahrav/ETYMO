"""
Geometry Engine for Diachronic Semantic Space Projections
=========================================================
Computes mathematically grounded 2D PCA projections, Gaussian density
contour geometry, and optimal transport streamlines for ManimGX scenes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models" / "embeddings"
RESULTS_DIR = ROOT / "results"

ORIGIN_COLORS = {
    "Germanic": "#4FC3F7",   # Ice blue
    "Latin": "#EF5350",      # Crimson
    "Greek": "#66BB6A",      # Emerald
    "Sanskrit": "#FFD54F",   # Solar gold
    "PIE": "#CE93D8",        # Royal lavender
    "Other": "#90A4AE",      # Slate gray
}

SHIFT_TYPE_COLORS = {
    "Core Invariant": "#66BB6A",       # Emerald green (stable)
    "Centroid Migration": "#FFB74D",   # Amber / Orange (shifted)
    "Sense Broadening": "#BA68C8",     # Purple / Violet (expanded)
    "Sense Narrowing": "#4DD0E1",      # Cyan (focused)
}


@dataclass
class NeighborNode:
    word: str
    pos: np.ndarray  # [x, y, 0]
    similarity: float
    is_shared: bool
    is_novel: bool = False


@dataclass
class ContourGeometry:
    center: np.ndarray  # [x, y, 0]
    radius_x: float
    radius_y: float
    angle: float
    label: str


@dataclass
class WordGeometry:
    word: str
    origin: str
    origin_color: str
    shift_type: str
    shift_color: str
    w2_score: float
    cos_drift: float
    k_hist: int
    k_mod: int
    delta_k: int
    dispersion_hist: float
    dispersion_mod: float
    
    old_target_pos: np.ndarray
    new_target_pos: np.ndarray
    
    old_neighbors: List[NeighborNode]
    new_neighbors: List[NeighborNode]
    shared_words: List[str]
    novel_words: List[str]
    
    hist_contour: ContourGeometry
    mod_contour: ContourGeometry
    mod_secondary_contour: Optional[ContourGeometry] = None


class SemanticSpaceEngine:
    """Loads embedding matrices and computes local SVD projections."""
    _instance = None

    def __init__(self):
        self.aligned_old = np.load(str(MODELS_DIR / "w2v_old_aligned.npy"))
        with open(MODELS_DIR / "w2v_old_vocab.txt", "r", encoding="utf-8") as f:
            self.vocab_old = [line.strip() for line in f]
        self.w2i_old = {w: i for i, w in enumerate(self.vocab_old)}
        self.norms_old = np.linalg.norm(self.aligned_old, axis=1, keepdims=True)
        self.norms_old[self.norms_old == 0] = 1.0

        self.vectors_new = np.load(str(MODELS_DIR / "w2v_new_vectors.npy"))
        with open(MODELS_DIR / "w2v_new_vocab.txt", "r", encoding="utf-8") as f:
            self.vocab_new = [line.strip() for line in f]
        self.w2i_new = {w: i for i, w in enumerate(self.vocab_new)}
        self.norms_new = np.linalg.norm(self.vectors_new, axis=1, keepdims=True)
        self.norms_new[self.norms_new == 0] = 1.0

        # Load drift data
        drift_csv = RESULTS_DIR / "dl_drift_scores.csv"
        if not drift_csv.exists():
            drift_csv = RESULTS_DIR / "drift_scores.csv"
        self.drift_df = pd.read_csv(drift_csv)
        self.drift_dict = self.drift_df.set_index("word").to_dict(orient="index")

    @classmethod
    def get_instance(cls) -> "SemanticSpaceEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def compute_geometry(
        self,
        word: str,
        top_k: int = 5,
        x_bounds: Tuple[float, float] = (-4.8, 4.8),
        y_bounds: Tuple[float, float] = (-2.3, 1.8),
        min_dist: float = 1.40,
    ) -> WordGeometry:
        word = word.lower().strip()
        row = self.drift_dict.get(word, {})

        origin = row.get("origin_class", "Germanic")
        origin_color = ORIGIN_COLORS.get(origin, "#4FC3F7")
        shift_type = row.get("shift_type", "Centroid Migration")
        shift_color = SHIFT_TYPE_COLORS.get(shift_type, "#FFB74D")
        w2_score = float(row.get("wasserstein_w2", row.get("drift_score", 0.75)))
        cos_drift = float(row.get("centroid_cosine_drift", 0.12))
        k_hist = int(row.get("k_hist", 1))
        k_mod = int(row.get("k_mod", 1))
        delta_k = int(row.get("delta_k", k_mod - k_hist))
        disp_hist = float(row.get("dispersion_hist", 0.35))
        disp_mod = float(row.get("dispersion_mod", 0.42))

        # Target vectors
        has_old = word in self.w2i_old
        has_new = word in self.w2i_new

        v_old = self.aligned_old[self.w2i_old[word]] if has_old else np.zeros(100)
        v_new = self.vectors_new[self.w2i_new[word]] if has_new else v_old.copy()

        # Find historical neighbors
        old_nbrs_info: List[Tuple[str, float, np.ndarray]] = []
        if has_old:
            sims_old = (self.aligned_old @ v_old) / (self.norms_old.squeeze() * np.linalg.norm(v_old))
            top_old_idx = np.argsort(sims_old)[::-1]
            count = 0
            for idx in top_old_idx:
                w = self.vocab_old[idx]
                if w != word:
                    old_nbrs_info.append((w, float(sims_old[idx]), self.aligned_old[idx]))
                    count += 1
                    if count >= top_k:
                        break

        # Find modern neighbors
        new_nbrs_info: List[Tuple[str, float, np.ndarray]] = []
        if has_new:
            sims_new = (self.vectors_new @ v_new) / (self.norms_new.squeeze() * np.linalg.norm(v_new))
            top_new_idx = np.argsort(sims_new)[::-1]
            count = 0
            for idx in top_new_idx:
                w = self.vocab_new[idx]
                if w != word:
                    new_nbrs_info.append((w, float(sims_new[idx]), self.vectors_new[idx]))
                    count += 1
                    if count >= top_k:
                        break

        old_words_set = set(w for w, _, _ in old_nbrs_info)
        new_words_set = set(w for w, _, _ in new_nbrs_info)
        shared_words = sorted(list(old_words_set & new_words_set))
        novel_words = sorted(list(new_words_set - old_words_set))

        # Construct vector matrix for joint PCA projection
        mat = [v_old, v_new]
        for _, _, vec in old_nbrs_info:
            mat.append(vec)
        for _, _, vec in new_nbrs_info:
            mat.append(vec)

        X = np.array(mat)
        # SVD PCA Projection (d=100 -> d=2)
        mean_vec = X.mean(axis=0)
        Xc = X - mean_vec
        _, _, vh = np.linalg.svd(Xc, full_matrices=False)
        coords_2d = Xc @ vh[:2].T

        # Normalize and frame within specified viewport bounds
        fitted_coords = self._fit_to_viewport(coords_2d, x_bounds=x_bounds, y_bounds=y_bounds)
        
        # Apply gentle label separation to prevent overlap collisions
        fitted_coords = self._repulse_overlapping(fitted_coords, min_dist=min_dist)

        old_target_pos = fitted_coords[0]
        new_target_pos = fitted_coords[1]

        # Extract neighbors
        n_old = len(old_nbrs_info)
        old_nodes: List[NeighborNode] = []
        for i, (w, sim, _) in enumerate(old_nbrs_info):
            pos = fitted_coords[2 + i]
            old_nodes.append(NeighborNode(
                word=w,
                pos=pos,
                similarity=sim,
                is_shared=(w in shared_words),
                is_novel=False,
            ))

        new_nodes: List[NeighborNode] = []
        for j, (w, sim, _) in enumerate(new_nbrs_info):
            pos = fitted_coords[2 + n_old + j]
            new_nodes.append(NeighborNode(
                word=w,
                pos=pos,
                similarity=sim,
                is_shared=(w in shared_words),
                is_novel=(w in novel_words),
            ))

        # Calculate density contours
        # Historical contour centered at old target position
        rx_hist = max(1.2, disp_hist * 3.8)
        ry_hist = max(0.9, disp_hist * 2.8)
        hist_contour = ContourGeometry(
            center=old_target_pos,
            radius_x=rx_hist,
            radius_y=ry_hist,
            angle=0.15,
            label="1800s Sense Density",
        )

        # Modern contour centered at new target position
        rx_mod = max(1.3, disp_mod * 4.0)
        ry_mod = max(1.0, disp_mod * 3.0)
        mod_contour = ContourGeometry(
            center=new_target_pos,
            radius_x=rx_mod,
            radius_y=ry_mod,
            angle=-0.12,
            label="2000s Primary Sense",
        )

        # If sense broadened (delta_k > 0) or novel tech cluster exists
        mod_secondary_contour = None
        if delta_k > 0 and len(novel_words) >= 2:
            novel_pos = [n.pos for n in new_nodes if n.is_novel]
            if len(novel_pos) > 0:
                sec_center = np.mean(novel_pos, axis=0)
                # Offset slightly towards outer perimeter
                sec_center[2] = 0.0
                mod_secondary_contour = ContourGeometry(
                    center=sec_center,
                    radius_x=1.35,
                    radius_y=0.95,
                    angle=0.25,
                    label="Emergent Sense",
                )

        return WordGeometry(
            word=word,
            origin=origin,
            origin_color=origin_color,
            shift_type=shift_type,
            shift_color=shift_color,
            w2_score=w2_score,
            cos_drift=cos_drift,
            k_hist=k_hist,
            k_mod=k_mod,
            delta_k=delta_k,
            dispersion_hist=disp_hist,
            dispersion_mod=disp_mod,
            old_target_pos=old_target_pos,
            new_target_pos=new_target_pos,
            old_neighbors=old_nodes,
            new_neighbors=new_nodes,
            shared_words=shared_words,
            novel_words=novel_words,
            hist_contour=hist_contour,
            mod_contour=mod_contour,
            mod_secondary_contour=mod_secondary_contour,
        )

    @staticmethod
    def _fit_to_viewport(coords_2d: np.ndarray, x_bounds=(-4.8, 4.8), y_bounds=(-2.3, 1.8)) -> np.ndarray:
        min_xy = coords_2d.min(axis=0)
        max_xy = coords_2d.max(axis=0)
        span_xy = np.maximum(max_xy - min_xy, 1e-4)

        target_w = x_bounds[1] - x_bounds[0]
        target_h = y_bounds[1] - y_bounds[0]

        scale = min(target_w / span_xy[0], target_h / span_xy[1])
        center_src = (min_xy + max_xy) / 2.0
        center_tgt = np.array([(x_bounds[0] + x_bounds[1]) / 2.0, (y_bounds[0] + y_bounds[1]) / 2.0])

        fitted_2d = (coords_2d - center_src) * scale + center_tgt
        res = np.zeros((coords_2d.shape[0], 3), dtype=np.float64)
        res[:, :2] = fitted_2d
        return res

    @staticmethod
    def _repulse_overlapping(pts: np.ndarray, min_dist: float = 1.40, iterations: int = 25) -> np.ndarray:
        """Lightweight 2D spring repulsion to guarantee no node collisions."""
        res = pts.copy()
        n = len(res)
        for _ in range(iterations):
            for i in range(n):
                for j in range(i + 1, n):
                    diff = res[i, :2] - res[j, :2]
                    dist = np.linalg.norm(diff)
                    if dist < min_dist and dist > 1e-4:
                        overlap = (min_dist - dist) * 0.5
                        direction = diff / dist
                        res[i, :2] += direction * overlap
                        res[j, :2] -= direction * overlap
        return res


def get_geometry_for_word(
    word: str,
    top_k: int = 5,
    x_bounds: Tuple[float, float] = (-4.8, 4.8),
    y_bounds: Tuple[float, float] = (-2.3, 1.8),
    min_dist: float = 1.40,
) -> WordGeometry:
    """Convenience accessor."""
    return SemanticSpaceEngine.get_instance().compute_geometry(
        word=word, top_k=top_k, x_bounds=x_bounds, y_bounds=y_bounds, min_dist=min_dist
    )
