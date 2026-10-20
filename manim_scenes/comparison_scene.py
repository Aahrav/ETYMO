"""
Scene 3: Two Words, Two Stories (Diachronic Semantic Comparison)
================================================================
Side-by-side comparative analysis of two words' semantic drift trajectories
using dual 2D PCA subspaces, Gaussian density contours, and synchronized
optimal transport streamlines.

Parametric scene:
    word_a: Primary word (e.g. "craft")
    word_b: Comparison word (e.g. "room" or "mouse")
"""

import sys
from pathlib import Path
import numpy as np
import manimgx as m

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from manim_scenes.geometry import get_geometry_for_word, WordGeometry
from manim_scenes.components import (
    GaussianDensityContour,
    TransportStreamline,
    NeighborNodeMobject,
    CosSimSpoke,
    EraBadge,
)

DEEP_BG = "#0B0E14"


class ComparisonCardHUD(m.VGroup):
    """Compact telemetry card for one half of the comparison screen."""
    def __init__(self, geo: WordGeometry, center_x: float, **kwargs):
        super().__init__(**kwargs)
        self.bg = m.RoundedRectangle(
            corner_radius=0.16,
            width=6.3,
            height=1.25,
            fill_color="#0D1117",
            fill_opacity=0.94,
            stroke_color="#30363D",
            stroke_width=1.4,
        )
        self.add(self.bg)

        # Row 1: Word Title + Origin pill
        title = m.Text(geo.word.upper(), font_size=26, color="#FFFFFF")
        
        origin_pill_bg = m.RoundedRectangle(
            corner_radius=0.08,
            width=max(1.8, len(geo.origin) * 0.18 + 0.6),
            height=0.42,
            fill_color=geo.origin_color,
            fill_opacity=0.22,
            stroke_color=geo.origin_color,
            stroke_width=1.2,
        )
        origin_pill_txt = m.Text(geo.origin, font_size=15, color=geo.origin_color)
        origin_pill_txt.move_to(origin_pill_bg.get_center())
        origin_pill = m.VGroup(origin_pill_bg, origin_pill_txt)

        row1 = m.VGroup(title, origin_pill).arrange(m.RIGHT, buff=0.25)
        row1.align_to(self.bg, m.LEFT).shift(m.RIGHT * 0.35 + m.UP * 0.22)

        # Row 2: Shift Archetype Pill + W2 Metric
        shift_pill_bg = m.RoundedRectangle(
            corner_radius=0.08,
            width=max(2.4, len(geo.shift_type) * 0.16 + 0.6),
            height=0.42,
            fill_color=geo.shift_color,
            fill_opacity=0.22,
            stroke_color=geo.shift_color,
            stroke_width=1.2,
        )
        shift_pill_txt = m.Text(geo.shift_type, font_size=15, color=geo.shift_color)
        shift_pill_txt.move_to(shift_pill_bg.get_center())
        shift_pill = m.VGroup(shift_pill_bg, shift_pill_txt)

        w2_badge = m.Text(f"W2 = {geo.w2_score:.4f}", font_size=16, color="#58A6FF")
        
        row2 = m.VGroup(shift_pill, w2_badge).arrange(m.RIGHT, buff=0.30)
        row2.align_to(self.bg, m.LEFT).shift(m.RIGHT * 0.35 + m.DOWN * 0.25)

        self.add(row1, row2)
        self.move_to(np.array([center_x, 3.15, 0.0]))


class ComparisonScene(m.Scene):
    word_a = "craft"
    word_b = "room"

    def construct(self) -> None:
        self.camera.background_color = DEEP_BG

        # Fetch dual viewport geometries
        geo_a: WordGeometry = get_geometry_for_word(
            self.word_a, x_bounds=(-5.4, -1.1), y_bounds=(-2.2, 1.8), min_dist=1.20
        )
        geo_b: WordGeometry = get_geometry_for_word(
            self.word_b, x_bounds=(1.1, 5.4), y_bounds=(-2.2, 1.8), min_dist=1.20
        )

        # ══════════════════════════════════════════════════════
        # ACT 0 — Dual HUDs, Divider & Atmosphere
        # ══════════════════════════════════════════════════════
        hud_a = ComparisonCardHUD(geo_a, center_x=-3.55)
        hud_b = ComparisonCardHUD(geo_b, center_x=3.55)

        divider = m.DashedLine(
            start=np.array([0.0, 2.4, 0.0]),
            end=np.array([0.0, -2.8, 0.0]),
            dash_length=0.18,
            dashed_ratio=0.55,
            stroke_color="#30363D",
            stroke_width=1.8,
        )

        vs_badge_bg = m.Circle(
            radius=0.30,
            fill_color="#0D1117",
            fill_opacity=1.0,
            stroke_color="#58A6FF",
            stroke_width=1.5,
        ).move_to(np.array([0.0, 3.15, 0.0]))
        vs_badge_txt = m.Text("VS", font_size=15, color="#58A6FF").move_to(vs_badge_bg.get_center())
        vs_badge = m.VGroup(vs_badge_bg, vs_badge_txt)

        era_badge = EraBadge(
            era_text="ERA: 1800 - 1900 (HISTORICAL CORPUS COMPARISON)",
            color="#58A6FF",
        )

        self.play(
            m.FadeIn(hud_a, shift=m.DOWN * 0.2),
            m.FadeIn(hud_b, shift=m.DOWN * 0.2),
            m.Create(divider),
            m.FadeIn(vs_badge),
            m.FadeIn(era_badge, shift=m.UP * 0.2),
            run_time=0.8,
        )

        # ══════════════════════════════════════════════════════
        # ACT I — Historical Constellations (Side by Side)
        # ══════════════════════════════════════════════════════
        def build_side(geo: WordGeometry):
            target_dot = m.Dot(point=geo.old_target_pos, radius=0.16, color=geo.origin_color)
            target_ring = m.Circle(
                radius=0.28, stroke_color=geo.origin_color, stroke_width=2.0, stroke_opacity=0.85, fill_opacity=0.0
            ).move_to(geo.old_target_pos)
            target_node = m.VGroup(target_ring, target_dot)

            contour = GaussianDensityContour(
                center=geo.hist_contour.center,
                radius_x=min(1.2, geo.hist_contour.radius_x * 0.70),
                radius_y=min(0.95, geo.hist_contour.radius_y * 0.70),
                angle=geo.hist_contour.angle,
                color=geo.origin_color,
            )

            nodes, spokes = [], []
            for nbr in geo.old_neighbors:
                node = NeighborNodeMobject(
                    word=nbr.word,
                    pos=nbr.pos,
                    center_pos=geo.old_target_pos,
                    color=geo.origin_color,
                    similarity=nbr.similarity,
                    is_shared=nbr.is_shared,
                )
                spoke = CosSimSpoke(
                    start_pt=geo.old_target_pos,
                    end_pt=nbr.pos,
                    similarity=nbr.similarity,
                    color=geo.origin_color,
                )
                nodes.append(node)
                spokes.append(spoke)

            return target_dot, target_ring, target_node, contour, m.VGroup(*nodes), m.VGroup(*spokes)

        dot_a, ring_a, target_a, contour_a, nodes_a, spokes_a = build_side(geo_a)
        dot_b, ring_b, target_b, contour_b, nodes_b, spokes_b = build_side(geo_b)

        self.play(
            m.FadeIn(contour_a, scale=0.88),
            m.FadeIn(contour_b, scale=0.88),
            m.Create(target_a),
            m.Create(target_b),
            run_time=1.0,
        )
        self.bring_to_front(target_a, target_b)

        self.play(
            m.Create(spokes_a),
            m.Create(spokes_b),
            m.FadeIn(nodes_a, scale=0.9),
            m.FadeIn(nodes_b, scale=0.9),
            run_time=1.2,
        )
        self.wait(1.0)

        # ══════════════════════════════════════════════════════
        # ACT II — Synchronized Diachronic Flight
        # ══════════════════════════════════════════════════════
        trans_badge = EraBadge(
            era_text="OPTIMAL TRANSPORT: SYNCHRONIZED VECTOR DRIFT",
            color="#BA68C8",
        )

        streamline_a = TransportStreamline(
            start_pt=geo_a.old_target_pos, end_pt=geo_a.new_target_pos, color="#BA68C8", arc_angle=-0.25
        )
        streamline_b = TransportStreamline(
            start_pt=geo_b.old_target_pos, end_pt=geo_b.new_target_pos, color="#BA68C8", arc_angle=-0.25
        )

        self.play(
            m.Transform(era_badge, trans_badge),
            m.Create(streamline_a),
            m.Create(streamline_b),
            m.FadeOut(spokes_a),
            m.FadeOut(spokes_b),
            m.FadeOut(contour_a),
            m.FadeOut(contour_b),
            m.FadeOut(nodes_a),
            m.FadeOut(nodes_b),
            run_time=1.0,
        )

        # Dual simultaneous target flight
        self.play(
            dot_a.animate.move_to(geo_a.new_target_pos),
            ring_a.animate.move_to(geo_a.new_target_pos),
            dot_b.animate.move_to(geo_b.new_target_pos),
            ring_b.animate.move_to(geo_b.new_target_pos),
            run_time=1.6,
        )

        # ══════════════════════════════════════════════════════
        # ACT III — Modern Constellations & Comparative Verdict
        # ══════════════════════════════════════════════════════
        def build_modern_side(geo: WordGeometry):
            contour = GaussianDensityContour(
                center=geo.mod_contour.center,
                radius_x=min(1.2, geo.mod_contour.radius_x * 0.70),
                radius_y=min(0.95, geo.mod_contour.radius_y * 0.70),
                angle=geo.mod_contour.angle,
                color="#66BB6A",
            )
            nodes, spokes = [], []
            for nbr in geo.new_neighbors:
                node = NeighborNodeMobject(
                    word=nbr.word,
                    pos=nbr.pos,
                    center_pos=geo.new_target_pos,
                    color="#66BB6A",
                    similarity=nbr.similarity,
                    is_shared=nbr.is_shared,
                    is_novel=nbr.is_novel,
                    is_historical=False,
                )
                spoke = CosSimSpoke(
                    start_pt=geo.new_target_pos,
                    end_pt=nbr.pos,
                    similarity=nbr.similarity,
                    color="#66BB6A",
                )
                nodes.append(node)
                spokes.append(spoke)
            return contour, m.VGroup(*nodes), m.VGroup(*spokes)

        mod_contour_a, mod_nodes_a, mod_spokes_a = build_modern_side(geo_a)
        mod_contour_b, mod_nodes_b, mod_spokes_b = build_modern_side(geo_b)

        delta_w2 = abs(geo_a.w2_score - geo_b.w2_score)
        more_stable = geo_a.word.upper() if geo_a.w2_score <= geo_b.w2_score else geo_b.word.upper()
        more_drifted = geo_b.word.upper() if geo_a.w2_score <= geo_b.w2_score else geo_a.word.upper()

        verdict_badge = EraBadge(
            era_text=f"VERDICT: {more_stable} MORE STABLE  |  {more_drifted} DRIFTED (ΔW2 = {delta_w2:.4f})",
            color="#FFB74D",
        )

        self.play(
            m.Transform(era_badge, verdict_badge),
            m.FadeIn(mod_contour_a, scale=0.88),
            m.FadeIn(mod_contour_b, scale=0.88),
            m.Create(mod_spokes_a),
            m.Create(mod_spokes_b),
            m.FadeIn(mod_nodes_a, scale=0.9),
            m.FadeIn(mod_nodes_b, scale=0.9),
            m.FadeOut(streamline_a),
            m.FadeOut(streamline_b),
            run_time=1.4,
        )
        self.bring_to_front(mod_nodes_a, mod_nodes_b, ring_a, dot_a, ring_b, dot_b)
        self.wait(1.5)


if __name__ == "__main__":
    print(f"ComparisonScene configured for: '{ComparisonScene.word_a}' vs '{ComparisonScene.word_b}'")
