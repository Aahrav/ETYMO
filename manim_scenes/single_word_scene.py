"""
Scene: Single Word Diachronic Semantic Shift (ManimGX)
======================================================
Mathematically grounded, publication-quality 3Blue1Brown style animation
visualizing a word's diachronic semantic drift using 2D PCA subspace
projections, Gaussian density contours, optimal transport streamlines,
and real-time telemetry HUD.

Default word: "craft"
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
    TelemetryHUD,
    NeighborNodeMobject,
    CosSimSpoke,
    EraBadge,
)

DEEP_BG = "#0B0E14"


class SingleWordDriftScene(m.Scene):
    target_word = "craft"

    def construct(self) -> None:
        self.camera.background_color = DEEP_BG

        # 1. Fetch mathematically computed 2D geometry
        geo: WordGeometry = get_geometry_for_word(self.target_word)

        # ══════════════════════════════════════════════════════
        # ACT 0 — Telemetry & Atmosphere Setup
        # ══════════════════════════════════════════════════════
        hud = TelemetryHUD(
            word=geo.word,
            origin=geo.origin,
            origin_color=geo.origin_color,
            shift_type=geo.shift_type,
            shift_color=geo.shift_color,
            w2_score=geo.w2_score,
            k_hist=geo.k_hist,
            k_mod=geo.k_mod,
        )

        era_badge = EraBadge(
            era_text="ERA: 1800 - 1900 (HISTORICAL GUTENBERG CORPUS)",
            color="#58A6FF",
        )

        self.play(
            m.FadeIn(hud, shift=m.DOWN * 0.2),
            m.FadeIn(era_badge, shift=m.UP * 0.2),
            run_time=0.8,
        )

        # ══════════════════════════════════════════════════════
        # ACT I — 1800s Historical Semantic Neighborhood
        # ══════════════════════════════════════════════════════
        # Historical target node (large glowing anchor)
        target_dot = m.Dot(
            point=geo.old_target_pos,
            radius=0.18,
            color=geo.origin_color,
        )
        target_ring = m.Circle(
            radius=0.30,
            stroke_color=geo.origin_color,
            stroke_width=2.2,
            stroke_opacity=0.85,
            fill_opacity=0.0,
        ).move_to(geo.old_target_pos)

        target_node_group = m.VGroup(target_ring, target_dot)

        # Gaussian density contour around 1800s senses
        hist_contour = GaussianDensityContour(
            center=geo.hist_contour.center,
            radius_x=geo.hist_contour.radius_x,
            radius_y=geo.hist_contour.radius_y,
            angle=geo.hist_contour.angle,
            color=geo.origin_color,
        )

        # Historical neighbor nodes and spokes
        old_nodes = []
        old_spokes = []
        for nbr in geo.old_neighbors:
            node = NeighborNodeMobject(
                word=nbr.word,
                pos=nbr.pos,
                center_pos=geo.old_target_pos,
                color=geo.origin_color,
                similarity=nbr.similarity,
                is_shared=nbr.is_shared,
                is_novel=False,
                is_historical=True,
            )
            spoke = CosSimSpoke(
                start_pt=geo.old_target_pos,
                end_pt=nbr.pos,
                similarity=nbr.similarity,
                color=geo.origin_color,
            )
            old_nodes.append(node)
            old_spokes.append(spoke)

        old_nodes_grp = m.VGroup(*old_nodes)
        old_spokes_grp = m.VGroup(*old_spokes)

        # Bloom historical constellation
        self.play(
            m.FadeIn(hist_contour, scale=0.88),
            m.Create(target_node_group),
            run_time=1.0,
        )
        self.bring_to_front(target_node_group)

        self.play(
            m.Create(old_spokes_grp),
            m.FadeIn(old_nodes_grp, scale=0.9),
            run_time=1.2,
        )
        self.wait(1.0)

        # ══════════════════════════════════════════════════════
        # ACT II — Optimal Transport & Semantic Transition
        # ══════════════════════════════════════════════════════
        # Update Era Badge to transition mode
        trans_era_badge = EraBadge(
            era_text=f"OPTIMAL TRANSPORT TRANSITION  |  W2 = {geo.w2_score:.4f}",
            color="#BA68C8",
        )

        streamline = TransportStreamline(
            start_pt=geo.old_target_pos,
            end_pt=geo.new_target_pos,
            color="#BA68C8",
            arc_angle=-0.25,
        )

        # Cleanly fade out historical spokes and nodes to prepare for flight
        self.play(
            m.Transform(era_badge, trans_era_badge),
            m.Create(streamline),
            m.FadeOut(old_spokes_grp),
            m.FadeOut(hist_contour),
            m.FadeOut(old_nodes_grp),
            run_time=1.0,
        )

        # Glide target node along the vector drift path to modern centroid
        self.play(
            target_dot.animate.move_to(geo.new_target_pos),
            target_ring.animate.move_to(geo.new_target_pos),
            run_time=1.6,
        )

        # ══════════════════════════════════════════════════════
        # ACT III — Modern Semantic Topology Emergence
        # ══════════════════════════════════════════════════════
        mod_era_badge = EraBadge(
            era_text="ERA: 2000 - 2020 (MODERN WIKIPEDIA CORPUS)",
            color="#66BB6A",
        )

        # Modern contour
        mod_contour = GaussianDensityContour(
            center=geo.mod_contour.center,
            radius_x=geo.mod_contour.radius_x,
            radius_y=geo.mod_contour.radius_y,
            angle=geo.mod_contour.angle,
            color="#66BB6A",
        )

        # Secondary contour if senses bifurcated / broadened
        sec_contour = None
        if geo.mod_secondary_contour:
            sec_contour = GaussianDensityContour(
                center=geo.mod_secondary_contour.center,
                radius_x=geo.mod_secondary_contour.radius_x,
                radius_y=geo.mod_secondary_contour.radius_y,
                angle=geo.mod_secondary_contour.angle,
                color="#BA68C8",
            )

        # Modern neighbor nodes and spokes
        new_nodes = []
        new_spokes = []
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
            new_nodes.append(node)
            new_spokes.append(spoke)

        new_nodes_grp = m.VGroup(*new_nodes)
        new_spokes_grp = m.VGroup(*new_spokes)

        # Animate emergence of modern constellation: contours first, then nodes in foreground
        anims = [
            m.Transform(era_badge, mod_era_badge),
            m.FadeIn(mod_contour, scale=0.88),
        ]
        if sec_contour:
            anims.append(m.FadeIn(sec_contour, scale=0.85))
        anims.extend([
            m.Create(new_spokes_grp),
            m.FadeIn(new_nodes_grp, scale=0.9),
            m.FadeOut(streamline),
        ])

        self.play(*anims, run_time=1.4)
        self.bring_to_front(new_nodes_grp, target_ring, target_dot)
        self.wait(1.0)

        # Highlight topology features: shared nodes pulse gold, novel pulse cyan
        shared_mobjects = [n for n in new_nodes if n.label.text in geo.shared_words]
        if shared_mobjects:
            self.play(
                *[m.Indicate(n.glow_ring, color="#FFD54F", scale_factor=1.4) for n in shared_mobjects],
                run_time=1.0,
            )

        novel_mobjects = [n for n in new_nodes if n.label.text in geo.novel_words[:3]]
        if novel_mobjects:
            self.play(
                *[m.Indicate(n.glow_ring, color="#4DD0E1", scale_factor=1.3) for n in novel_mobjects],
                run_time=1.0,
            )

        self.wait(1.2)


if __name__ == "__main__":
    print(f"SingleWordDriftScene configured for word: '{SingleWordDriftScene.target_word}'")
