"""
Scene 1: What Are Word Embeddings? (Educational Introduction)
==============================================================
3Blue1Brown-quality pedagogical animation explaining word vectors,
vector arithmetic (king - man + woman = queen), high-dimensional geometry,
and setting up the core concept of diachronic semantic drift.
"""

import sys
from pathlib import Path
import numpy as np
import manimgx as m

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from manim_scenes.components import EraBadge

DEEP_BG = "#0B0E14"


class EmbeddingsIntroHUD(m.VGroup):
    """Top telemetry bar for the intro explainer."""
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

        title = m.Text("WHAT ARE WORD EMBEDDINGS?", font_size=24, color="#FFFFFF")
        title.align_to(self.bg, m.LEFT).shift(m.RIGHT * 0.40)

        subtitle = m.Text("DISTRIBUTIONAL SEMANTICS: d = 100", font_size=15, color="#58A6FF")
        subtitle.align_to(self.bg, m.RIGHT).shift(m.LEFT * 0.40)

        self.add(title, subtitle)
        self.move_to(m.UP * 3.20)


class EmbeddingsIntroScene(m.Scene):
    def construct(self) -> None:
        self.camera.background_color = DEEP_BG

        # ══════════════════════════════════════════════════════
        # ACT 0 — Atmospheric Setup & HUD
        # ══════════════════════════════════════════════════════
        hud = EmbeddingsIntroHUD()
        era_badge = EraBadge(
            era_text="CORE CONCEPT: WORDS AS POINTS IN HIGH-DIMENSIONAL CONTINUOUS SPACE",
            color="#58A6FF",
        )

        self.play(
            m.FadeIn(hud, shift=m.DOWN * 0.2),
            m.FadeIn(era_badge, shift=m.UP * 0.2),
            run_time=0.8,
        )

        # ══════════════════════════════════════════════════════
        # ACT I — Word to Vector Transformation
        # ══════════════════════════════════════════════════════
        word_card_bg = m.RoundedRectangle(
            corner_radius=0.14,
            width=3.2,
            height=1.0,
            fill_color="#1F242C",
            fill_opacity=0.9,
            stroke_color="#4FC3F7",
            stroke_width=1.5,
        ).move_to(np.array([0.0, 0.5, 0.0]))
        word_card_txt = m.Text('"king"', font_size=28, color="#4FC3F7").move_to(word_card_bg.get_center())
        word_card = m.VGroup(word_card_bg, word_card_txt)

        self.play(m.FadeIn(word_card, scale=0.85), run_time=0.8)
        self.wait(0.6)

        vec_card_bg = m.RoundedRectangle(
            corner_radius=0.14,
            width=8.4,
            height=1.0,
            fill_color="#1F242C",
            fill_opacity=0.9,
            stroke_color="#BA68C8",
            stroke_width=1.5,
        ).move_to(np.array([0.0, 0.5, 0.0]))
        vec_card_txt = m.Text("v_king = [ +0.28, -0.42, +0.91, ... ]  ∈ ℝ¹⁰⁰", font_size=22, color="#E6EDF3").move_to(vec_card_bg.get_center())
        vec_card = m.VGroup(vec_card_bg, vec_card_txt)

        self.play(
            m.Transform(word_card, vec_card),
            run_time=1.0,
        )
        self.wait(1.0)
        self.play(m.FadeOut(word_card), run_time=0.6)

        # ══════════════════════════════════════════════════════
        # ACT II — 2D Projection: Semantic Clustering & Arithmetic
        # ══════════════════════════════════════════════════════
        arith_badge = EraBadge(
            era_text="VECTOR ARITHMETIC: v_king - v_man + v_woman ≈ v_queen",
            color="#BA68C8",
        )

        # Coordinate axes frame
        axes_box = m.RoundedRectangle(
            corner_radius=0.16,
            width=10.0,
            height=4.6,
            fill_color="#0D1117",
            fill_opacity=0.70,
            stroke_color="#30363D",
            stroke_width=1.4,
        ).move_to(np.array([0.0, -0.25, 0.0]))

        # Illustrative coordinates
        pts = {
            "king":  np.array([2.0, 1.1, 0.0]),
            "queen": np.array([2.4, -0.1, 0.0]),
            "man":   np.array([-1.2, 0.8, 0.0]),
            "woman": np.array([-0.8, -0.4, 0.0]),
            "dog":   np.array([-3.4, -1.2, 0.0]),
            "cat":   np.array([-2.4, -1.6, 0.0]),
        }

        dots = {}
        for w, p in pts.items():
            col = "#4FC3F7" if w in ["king", "queen", "man", "woman"] else "#66BB6A"
            dot = m.Dot(point=p, radius=0.13, color=col)
            lbl = m.Text(w, font_size=16, color=col)
            lbl.next_to(dot, m.UP, buff=0.20)
            lbl.add_background_rectangle(color="#0B0E14", opacity=0.92, buff=0.08)
            dots[w] = m.VGroup(dot, lbl)

        # Arrows demonstrating vector arithmetic
        # Vector from man to king
        arrow_man_king = m.Arrow(
            start=pts["man"], end=pts["king"],
            stroke_color="#BA68C8", stroke_width=2.5, buff=0.20,
        )
        # Vector from woman to queen
        arrow_woman_queen = m.Arrow(
            start=pts["woman"], end=pts["queen"],
            stroke_color="#BA68C8", stroke_width=2.5, buff=0.20,
        )

        self.play(
            m.Transform(era_badge, arith_badge),
            m.Create(axes_box),
            *[m.FadeIn(dots[w]) for w in pts],
            run_time=1.2,
        )
        self.play(
            m.Create(arrow_man_king),
            m.Create(arrow_woman_queen),
            run_time=1.2,
        )
        self.wait(1.5)

        # ══════════════════════════════════════════════════════
        # ACT III — Diachronic Question: What If Words Shift?
        # ══════════════════════════════════════════════════════
        drift_intro_badge = EraBadge(
            era_text="DIACHRONIC QUESTION: HOW DO EMBEDDINGS EVOLVE ACROSS CENTURIES?",
            color="#FFD54F",
        )

        self.play(
            m.Transform(era_badge, drift_intro_badge),
            m.FadeOut(arrow_man_king),
            m.FadeOut(arrow_woman_queen),
            *[m.FadeOut(dots[w]) for w in ["dog", "cat", "king", "queen", "man", "woman"]],
            run_time=1.0,
        )

        # Mouse: 1800s vs 2000s
        pos_old = np.array([-2.6, 0.4, 0.0])
        pos_new = np.array([2.6, -0.4, 0.0])

        mouse_dot = m.Dot(point=pos_old, radius=0.16, color="#4FC3F7")
        mouse_ring = m.Circle(radius=0.28, stroke_color="#4FC3F7", stroke_width=2.0, fill_opacity=0.0).move_to(pos_old)
        mouse_lbl = m.Text("mouse (1800s)", font_size=16, color="#FFFFFF")
        mouse_lbl.next_to(mouse_ring, m.UP, buff=0.22)
        mouse_lbl.add_background_rectangle(color="#0B0E14", opacity=0.92, buff=0.08)
        mouse_group = m.VGroup(mouse_ring, mouse_dot, mouse_lbl)

        # 1800s rodent neighbors
        old_nbr1 = m.Text("Rodent", font_size=18, color="#8B949E").move_to(pos_old + np.array([-1.2, -0.7, 0.0]))
        old_nbr2 = m.Text("Trap", font_size=18, color="#8B949E").move_to(pos_old + np.array([1.1, 0.7, 0.0]))
        old_nbrs = m.VGroup(old_nbr1, old_nbr2)

        self.play(
            m.Create(mouse_group),
            m.FadeIn(old_nbrs),
            run_time=1.0,
        )
        self.wait(1.0)

        # Glide to modern technology cluster
        streamline = m.ArcBetweenPoints(
            start=pos_old, end=pos_new, angle=-0.25,
            stroke_color="#BA68C8", stroke_width=3.0,
        )

        mouse_lbl_new = m.Text("mouse (2000s)", font_size=16, color="#FFFFFF")
        mouse_lbl_new.next_to(m.Circle(radius=0.28).move_to(pos_new), m.UP, buff=0.22)
        mouse_lbl_new.add_background_rectangle(color="#0B0E14", opacity=0.92, buff=0.08)

        new_nbr1 = m.Text("Cursor", font_size=18, color="#4DD0E1").move_to(pos_new + np.array([-1.1, -0.7, 0.0]))
        new_nbr2 = m.Text("Laptop", font_size=18, color="#4DD0E1").move_to(pos_new + np.array([1.1, -0.7, 0.0]))
        new_nbrs = m.VGroup(new_nbr1, new_nbr2)

        drift_verdict_badge = EraBadge(
            era_text="DIACHRONIC SEMANTIC SHIFT: MEANINGS MIGRATE ACROSS GEODESIC CURVES",
            color="#66BB6A",
        )

        self.play(
            m.Transform(era_badge, drift_verdict_badge),
            m.Create(streamline),
            m.FadeOut(old_nbrs),
            m.FadeOut(mouse_lbl),
            mouse_dot.animate.move_to(pos_new),
            mouse_ring.animate.move_to(pos_new),
            run_time=1.8,
        )
        self.play(
            m.FadeIn(mouse_lbl_new),
            m.FadeIn(new_nbrs),
            run_time=1.0,
        )
        self.wait(1.8)


if __name__ == "__main__":
    print("EmbeddingsIntroScene configured for educational intro.")
