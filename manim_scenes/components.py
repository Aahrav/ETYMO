"""
ManimGX Visual Components for Semantic Drift
============================================
Defines reusable, mathematically grounded visual mobjects:
- GaussianDensityContour (concentric probability density ellipses)
- TransportStreamline (geodesic Wasserstein transport curve with halo)
- TelemetryHUD (sleek top telemetry bar with etymology & drift metrics)
- NeighborNodeMobject (glowing vector nodes with collision-safe labels)
- CosSimSpoke (delicate cosine similarity connectors)
- EraBadge (subtle animated timeline badge)
"""

import numpy as np
import manimgx as m
from typing import Optional, List, Tuple


class GaussianDensityContour(m.VGroup):
    """Visualizes multi-sense Gaussian density contours p(x) ~ N(mu, Sigma)."""
    def __init__(
        self,
        center: np.ndarray,
        radius_x: float,
        radius_y: float,
        angle: float = 0.0,
        color: str = "#4FC3F7",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Concentric ellipses with fading alpha for soft glow
        self.outer = m.Ellipse(
            width=radius_x * 2.0,
            height=radius_y * 2.0,
            fill_color=color,
            fill_opacity=0.07,
            stroke_color=color,
            stroke_opacity=0.25,
            stroke_width=0.8,
        )
        self.mid = m.Ellipse(
            width=radius_x * 1.35,
            height=radius_y * 1.35,
            fill_color=color,
            fill_opacity=0.12,
            stroke_color=color,
            stroke_opacity=0.0,
            stroke_width=0.0,
        )
        self.inner = m.Ellipse(
            width=radius_x * 0.75,
            height=radius_y * 0.75,
            fill_color=color,
            fill_opacity=0.20,
            stroke_color=color,
            stroke_opacity=0.0,
            stroke_width=0.0,
        )
        self.add(self.outer, self.mid, self.inner)
        self.move_to(center)
        if angle != 0:
            self.rotate(angle)


class TransportStreamline(m.VGroup):
    """Optimal transport streamline connecting old and new semantic centroids."""
    def __init__(
        self,
        start_pt: np.ndarray,
        end_pt: np.ndarray,
        color: str = "#BB86FC",
        arc_angle: float = -0.28,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Glow halo
        self.halo = m.ArcBetweenPoints(
            start=start_pt,
            end=end_pt,
            angle=arc_angle,
            stroke_color=color,
            stroke_opacity=0.25,
            stroke_width=5.5,
        )
        # Core streamline
        self.core = m.ArcBetweenPoints(
            start=start_pt,
            end=end_pt,
            angle=arc_angle,
            stroke_color=color,
            stroke_opacity=0.90,
            stroke_width=2.5,
        )
        self.add(self.halo, self.core)


class TelemetryHUD(m.VGroup):
    """Upper telemetry HUD displaying target word, etymology origin, and drift metrics."""
    def __init__(
        self,
        word: str,
        origin: str,
        origin_color: str,
        shift_type: str,
        shift_color: str,
        w2_score: float,
        k_hist: int,
        k_mod: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Glassmorphic backplate
        self.bg = m.RoundedRectangle(
            corner_radius=0.18,
            width=13.4,
            height=1.20,
            fill_color="#0D1117",
            fill_opacity=0.92,
            stroke_color="#30363D",
            stroke_width=1.5,
        )
        self.add(self.bg)

        # Target Word
        self.title_word = m.Text(
            word.upper(),
            font_size=28,
            color="#FFFFFF",
        )
        self.title_word.align_to(self.bg, m.LEFT).shift(m.RIGHT * 0.45)

        # Etymology Pill
        origin_pill_bg = m.RoundedRectangle(
            corner_radius=0.10,
            width=max(1.8, len(origin) * 0.18 + 0.6),
            height=0.46,
            fill_color=origin_color,
            fill_opacity=0.22,
            stroke_color=origin_color,
            stroke_width=1.2,
        )
        origin_pill_txt = m.Text(
            origin,
            font_size=15,
            color=origin_color,
        )
        self.origin_pill = m.VGroup(origin_pill_bg, origin_pill_txt)
        origin_pill_txt.move_to(origin_pill_bg.get_center())
        self.origin_pill.next_to(self.title_word, m.RIGHT, buff=0.30)

        # Shift Archetype Pill
        shift_pill_bg = m.RoundedRectangle(
            corner_radius=0.10,
            width=max(2.4, len(shift_type) * 0.16 + 0.6),
            height=0.46,
            fill_color=shift_color,
            fill_opacity=0.22,
            stroke_color=shift_color,
            stroke_width=1.2,
        )
        shift_pill_txt = m.Text(
            shift_type,
            font_size=15,
            color=shift_color,
        )
        self.shift_pill = m.VGroup(shift_pill_bg, shift_pill_txt)
        shift_pill_txt.move_to(shift_pill_bg.get_center())
        self.shift_pill.next_to(self.origin_pill, m.RIGHT, buff=0.25)

        # Metric Badges (Right side)
        k_str = f"Senses: k={k_hist} -> {k_mod}" if k_hist != k_mod else f"Senses: k={k_hist}"
        self.k_badge = m.Text(
            k_str,
            font_size=16,
            color="#8B949E",
        )

        # Wasserstein W2 Drift Metric
        self.w2_badge = m.Text(
            f"Wasserstein W2: {w2_score:.4f}",
            font_size=17,
            color="#58A6FF",
        )

        metrics_group = m.VGroup(self.w2_badge, self.k_badge).arrange(m.DOWN, buff=0.08, aligned_edge=m.RIGHT)
        metrics_group.align_to(self.bg, m.RIGHT).shift(m.LEFT * 0.45)

        self.add(self.title_word, self.origin_pill, self.shift_pill, metrics_group)
        self.move_to(m.UP * 3.15)


class NeighborNodeMobject(m.VGroup):
    """Individual neighbor node in the 2D projected semantic space."""
    def __init__(
        self,
        word: str,
        pos: np.ndarray,
        center_pos: np.ndarray,
        color: str,
        similarity: float,
        is_shared: bool = False,
        is_novel: bool = False,
        is_historical: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_pos = pos

        # Node dot with ring
        node_radius = 0.11 if not is_shared else 0.14
        ring_color = "#FFD54F" if is_shared else ("#4DD0E1" if is_novel else color)
        
        self.glow_ring = m.Circle(
            radius=node_radius + 0.08,
            stroke_color=ring_color,
            stroke_opacity=0.75,
            stroke_width=1.5,
            fill_opacity=0.0,
        )
        self.dot = m.Dot(
            point=pos,
            radius=node_radius,
            color=ring_color,
        )
        self.glow_ring.move_to(pos)

        # Word label pointing outward away from target center
        label_color = "#FFFFFF" if (is_shared or is_novel) else "#E6EDF3"
        self.label = m.Text(
            word,
            font_size=15,
            color=label_color,
        )
        
        # Outward direction vector
        diff = pos[:2] - center_pos[:2]
        dist = np.linalg.norm(diff)
        if dist > 1e-4:
            outward = np.array([diff[0] / dist, diff[1] / dist, 0.0])
        else:
            outward = m.UP
        
        self.label.move_to(pos + outward * 0.38)
        self.label.add_background_rectangle(color="#0B0E14", opacity=0.92, buff=0.08)

        self.add(self.glow_ring, self.dot, self.label)


class CosSimSpoke(m.VGroup):
    """Delicate line connecting target word to a neighbor with trimmed ends."""
    def __init__(
        self,
        start_pt: np.ndarray,
        end_pt: np.ndarray,
        similarity: float,
        color: str = "#8B949E",
        target_radius: float = 0.34,
        nbr_radius: float = 0.22,
        **kwargs,
    ):
        super().__init__(**kwargs)
        diff = end_pt[:2] - start_pt[:2]
        dist = np.linalg.norm(diff)
        if dist > (target_radius + nbr_radius + 0.1):
            unit_dir = np.array([diff[0] / dist, diff[1] / dist, 0.0])
            trimmed_start = start_pt + unit_dir * target_radius
            trimmed_end = end_pt - unit_dir * nbr_radius
        else:
            trimmed_start = start_pt
            trimmed_end = end_pt

        self.spoke_line = m.Line(
            start=trimmed_start,
            end=trimmed_end,
            stroke_color=color,
            stroke_opacity=0.28,
            stroke_width=1.2,
        )
        self.add(self.spoke_line)


class EraBadge(m.VGroup):
    """Bottom badge indicating current historical, transition, or comparison verdict era."""
    def __init__(
        self,
        era_text: str = "ERA: 1800 - 1900 (HISTORICAL CORPUS)",
        color: str = "#58A6FF",
        font_size: int = 15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.txt = m.Text(
            era_text,
            font_size=font_size,
            color=color,
        )
        badge_w = max(6.4, self.txt.width + 0.8)
        badge_h = max(0.50, self.txt.height + 0.22)
        self.bg = m.RoundedRectangle(
            corner_radius=0.12,
            width=badge_w,
            height=badge_h,
            fill_color="#0D1117",
            fill_opacity=0.92,
            stroke_color=color,
            stroke_width=1.2,
        )
        self.txt.move_to(self.bg.get_center())
        self.add(self.bg, self.txt)
        self.move_to(m.DOWN * 3.35)
