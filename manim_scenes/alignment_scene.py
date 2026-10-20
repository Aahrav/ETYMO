"""
Scene 5: The Procrustes Alignment (Mathematical Method Explainer)
==================================================================
Visually explains why separate Word2Vec spaces cannot be compared directly
due to stochastic basis rotations, and how Orthogonal Procrustes alignment
solves the problem via SVD (W* = U V^T) using stable anchor words.
"""

import sys
from pathlib import Path
import numpy as np
import manimgx as m

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from manim_scenes.components import EraBadge

DEEP_BG = "#0B0E14"


class AlignmentHUD(m.VGroup):
    """Top telemetry bar for the Procrustes alignment explainer."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bg = m.RoundedRectangle(
            corner_radius=0.16,
            width=13.4,
            height=1.15,
            fill_color="#0D1117",
            fill_opacity=0.92,
            stroke_color="#30363D",
            stroke_width=1.4,
        )
        self.add(self.bg)

        # Title
        title = m.Text("ORTHOGONAL PROCRUSTES ALIGNMENT", font_size=24, color="#FFFFFF")
        title.align_to(self.bg, m.LEFT).shift(m.RIGHT * 0.40)

        # Formula badge
        formula = m.Text("CLOSED FORM: W* = U V^T", font_size=15, color="#BA68C8")
        formula.align_to(self.bg, m.RIGHT).shift(m.LEFT * 0.40)

        self.add(title, formula)
        self.move_to(m.UP * 3.20)


class AlignmentScene(m.Scene):
    def construct(self) -> None:
        self.camera.background_color = DEEP_BG

        # ══════════════════════════════════════════════════════
        # ACT 0 — Atmosphere & HUD
        # ══════════════════════════════════════════════════════
        hud = AlignmentHUD()
        era_badge = EraBadge(
            era_text="STEP 1: TWO INDEPENDENTLY TRAINED EMBEDDING SPACES",
            color="#58A6FF",
        )

        self.play(
            m.FadeIn(hud, shift=m.DOWN * 0.2),
            m.FadeIn(era_badge, shift=m.UP * 0.2),
            run_time=0.8,
        )

        # ══════════════════════════════════════════════════════
        # ACT I — The Dual Unaligned Spaces (Left vs Right)
        # ══════════════════════════════════════════════════════
        # Left frame: 1800s Space
        box_left = m.RoundedRectangle(
            corner_radius=0.14,
            width=5.8,
            height=4.4,
            fill_color="#0D1117",
            fill_opacity=0.75,
            stroke_color="#30363D",
            stroke_width=1.2,
        ).shift(m.LEFT * 3.4 + m.DOWN * 0.2)

        lbl_left = m.Text("1800s Coordinate Space (A)", font_size=15, color="#58A6FF")
        lbl_left.next_to(box_left, m.UP, buff=0.15)

        # Right frame: 2000s Space
        box_right = m.RoundedRectangle(
            corner_radius=0.14,
            width=5.8,
            height=4.4,
            fill_color="#0D1117",
            fill_opacity=0.75,
            stroke_color="#30363D",
            stroke_width=1.2,
        ).shift(m.RIGHT * 3.4 + m.DOWN * 0.2)

        lbl_right = m.Text("2000s Coordinate Space (B)", font_size=15, color="#EF5350")
        lbl_right.next_to(box_right, m.UP, buff=0.15)

        self.play(
            m.Create(box_left),
            m.Create(box_right),
            m.FadeIn(lbl_left),
            m.FadeIn(lbl_right),
            run_time=1.0,
        )

        # Words in 1800s Space
        # Anchors (gold): water, sun, stone; Target (blue/purple): mouse
        words_old = {
            "water": np.array([-4.5, 0.8, 0.0]),
            "sun":   np.array([-3.2, 1.2, 0.0]),
            "stone": np.array([-2.2, -0.6, 0.0]),
            "mouse": np.array([-4.0, -1.2, 0.0]),
        }

        # The SAME semantic words in 2000s space, but arbitrarily rotated by ~110 degrees!
        # Rotation + reflection representing stochastic initialization
        words_new_unaligned = {
            "water": np.array([2.5, -0.9, 0.0]),
            "sun":   np.array([3.4, -1.4, 0.0]),
            "stone": np.array([4.6, 0.4, 0.0]),
            "mouse": np.array([2.2, 1.0, 0.0]),
        }

        # Target aligned positions in Right box (matching left geometry, with mouse drifted!)
        words_new_aligned = {
            "water": np.array([2.3, 0.8, 0.0]),   # Perfect anchor match (offset by +6.8)
            "sun":   np.array([3.6, 1.2, 0.0]),   # Perfect anchor match
            "stone": np.array([4.6, -0.6, 0.0]),  # Perfect anchor match
            "mouse": np.array([4.8, 0.8, 0.0]),   # TRUE DRIFT: moved to modern tech sector!
        }

        dots_left = {}
        dots_right = {}

        for w, pos in words_old.items():
            col = "#FFD54F" if w != "mouse" else "#4FC3F7"
            dot = m.Dot(point=pos, radius=0.12, color=col)
            lbl = m.Text(w, font_size=15, color=col)
            lbl.next_to(dot, m.UP, buff=0.22)
            lbl.add_background_rectangle(color="#0B0E14", opacity=0.92, buff=0.08)
            dots_left[w] = m.VGroup(dot, lbl)

        for w, pos in words_new_unaligned.items():
            col = "#FFD54F" if w != "mouse" else "#EF5350"
            dot = m.Dot(point=pos, radius=0.12, color=col)
            lbl = m.Text(w, font_size=15, color=col)
            lbl.next_to(dot, m.UP, buff=0.22)
            lbl.add_background_rectangle(color="#0B0E14", opacity=0.92, buff=0.08)
            dots_right[w] = m.VGroup(dot, lbl)

        self.play(
            *[m.FadeIn(dots_left[w]) for w in words_old],
            *[m.FadeIn(dots_right[w]) for w in words_new_unaligned],
            run_time=1.2,
        )
        self.wait(1.0)

        # ══════════════════════════════════════════════════════
        # ACT II — The Dilemma: Stochastic Coordinate Basis
        # ══════════════════════════════════════════════════════
        mismatch_badge = EraBadge(
            era_text="PROBLEM: ARBITRARY ROTATION -- DIRECT DISTANCES ARE INVALID",
            color="#EF5350",
        )

        # Connect mismatched pairs with red dashed lines
        mismatch_lines = []
        for w in words_old:
            p1 = dots_left[w][0].get_center()
            p2 = dots_right[w][0].get_center()
            line = m.DashedLine(
                start=p1,
                end=p2,
                stroke_color="#EF5350",
                stroke_opacity=0.35,
                stroke_width=1.2,
            )
            mismatch_lines.append(line)

        mismatch_grp = m.VGroup(*mismatch_lines)

        self.play(
            m.Transform(era_badge, mismatch_badge),
            m.Create(mismatch_grp),
            run_time=1.2,
        )
        self.wait(1.5)

        # ══════════════════════════════════════════════════════
        # ACT III — Orthogonal Procrustes Solution (SVD Rotation)
        # ══════════════════════════════════════════════════════
        procrustes_badge = EraBadge(
            era_text="SOLUTION: ORTHOGONAL PROCRUSTES (M = A^T B = U Σ V^T  ==>  W* = U V^T)",
            color="#BA68C8",
        )

        self.play(
            m.Transform(era_badge, procrustes_badge),
            m.FadeOut(mismatch_grp),
            run_time=1.0,
        )

        # Rotate right dots to their aligned coordinates
        align_anims = []
        for w in words_old:
            target_pt = words_new_aligned[w]
            align_anims.append(dots_right[w][0].animate.move_to(target_pt))
            align_anims.append(dots_right[w][1].animate.move_to(target_pt + m.UP * 0.45))

        self.play(*align_anims, run_time=2.0)
        self.wait(1.0)

        # ══════════════════════════════════════════════════════
        # ACT IV — Verification: Anchors Aligned, Drift Revealed
        # ══════════════════════════════════════════════════════
        verdict_badge = EraBadge(
            era_text="ALIGNED: ANCHORS OVERLAP (Δ ≈ 0)  |  'MOUSE' REVEALS TRUE DIACHRONIC DRIFT",
            color="#66BB6A",
        )

        # Highlight anchor alignment with green checks
        anchor_rings = []
        for w in ["water", "sun", "stone"]:
            ring = m.Circle(
                radius=0.22,
                stroke_color="#66BB6A",
                stroke_width=2.0,
                fill_opacity=0.0,
            ).move_to(words_new_aligned[w])
            anchor_rings.append(ring)

        # Highlight mouse drift vector
        mouse_callout_ring = m.Circle(
            radius=0.24,
            stroke_color="#BA68C8",
            stroke_width=2.2,
            fill_opacity=0.0,
        ).move_to(words_new_aligned["mouse"])

        self.play(
            m.Transform(era_badge, verdict_badge),
            *[m.Create(r) for r in anchor_rings],
            m.Create(mouse_callout_ring),
            run_time=1.2,
        )

        self.play(
            *[m.Indicate(r, color="#66BB6A", scale_factor=1.3) for r in anchor_rings],
            m.Indicate(mouse_callout_ring, color="#BA68C8", scale_factor=1.4),
            run_time=1.2,
        )
        self.wait(1.5)


if __name__ == "__main__":
    print("AlignmentScene configured for Orthogonal Procrustes explainer.")
