"""
Scene 3: "Two Words, Two Stories" (Comparison Animation)
==========================================================
Deep-dive comparison of two words — shows their neighbor constellations
in both time periods side by side.

This scene is PARAMETRIC — accepts word_a and word_b as config variables
so the web app can render any pair on demand.

Default comparison: "craft" vs "room"
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    W2V_MODEL_OLD, W2V_MODEL_NEW, W2V_ALIGNED_OLD,
    RESULTS_DIR,
)

DEEP_BG = "#1a1a2e"
GERMANIC_BLUE = "#4FC3F7"
LATIN_RED = "#EF5350"
GREEK_GREEN = "#66BB6A"
OTHER_AMBER = "#FFA726"
SOFT_WHITE = "#e0e0e0"
ACCENT_PURPLE = "#BB86FC"
GRID_DIM = "#2a2a4e"

ORIGIN_COLORS = {
    "Germanic": GERMANIC_BLUE,
    "Latin": LATIN_RED,
    "Greek": GREEK_GREEN,
    "Other": OTHER_AMBER,
}


def load_data():
    """Load models, aligned vectors, and drift data."""
    from gensim.models import Word2Vec

    model_old = Word2Vec.load(str(W2V_MODEL_OLD))
    model_new = Word2Vec.load(str(W2V_MODEL_NEW))
    aligned_old = np.load(str(W2V_ALIGNED_OLD))

    old_words = list(model_old.wv.key_to_index.keys())
    old_w2i = {w: i for i, w in enumerate(old_words)}

    drift_df = pd.read_csv(RESULTS_DIR / "drift_scores.csv")

    return model_old, model_new, aligned_old, old_words, old_w2i, drift_df


def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


class ComparisonScene(Scene):
    # Override these for different word pairs
    word_a = "craft"
    word_b = "room"

    def construct(self):
        self.camera.background_color = DEEP_BG

        model_old, model_new, aligned_old, old_words, old_w2i, drift_df = load_data()

        # Get word info
        def get_word_info(word):
            row = drift_df[drift_df["word"] == word]
            if len(row) == 0:
                return {"origin": "Unknown", "drift": 0, "status": "MISSING"}
            row = row.iloc[0]
            return {
                "origin": row["origin_class"],
                "drift": row["drift_score"] if row["status"] == "OK" else 0,
                "status": row["status"],
            }

        def get_old_neighbors(word, topn=5):
            if word not in old_w2i:
                return []
            target = aligned_old[old_w2i[word]]
            norms = np.linalg.norm(aligned_old, axis=1)
            norms[norms == 0] = 1
            sims = aligned_old @ target / (norms * np.linalg.norm(target))
            top_idxs = np.argsort(sims)[::-1][1:topn + 1]
            return [(old_words[i], float(sims[i])) for i in top_idxs]

        info_a = get_word_info(self.word_a)
        info_b = get_word_info(self.word_b)

        color_a = ORIGIN_COLORS.get(info_a["origin"], SOFT_WHITE)
        color_b = ORIGIN_COLORS.get(info_b["origin"], SOFT_WHITE)

        # ── Shot 1: Title ──
        title = Text(
            f'Comparing: "{self.word_a}" vs "{self.word_b}"',
            font_size=38, color=SOFT_WHITE,
        )
        origin_badges = VGroup(
            Text(f"Origin: {info_a['origin']}", font_size=18, color=color_a),
            Text("  vs  ", font_size=18, color=SOFT_WHITE),
            Text(f"Origin: {info_b['origin']}", font_size=18, color=color_b),
        ).arrange(RIGHT).next_to(title, DOWN, buff=0.4)

        self.play(Write(title, run_time=1.5))
        self.play(FadeIn(origin_badges))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(origin_badges))

        # ── Shot 2: Split screen ──
        divider = Line(
            [0, -3.5, 0], [0, 3.5, 0],
            color=GRID_DIM, stroke_width=1,
        )
        self.play(Create(divider))

        label_a = Text(f'"{self.word_a}"', font_size=28, color=color_a).move_to([-3.5, 3.2, 0])
        label_b = Text(f'"{self.word_b}"', font_size=28, color=color_b).move_to([3.5, 3.2, 0])
        self.play(FadeIn(label_a), FadeIn(label_b))

        # ── Shot 3: 1800s neighbor constellations ──
        era_old = Text("1800s", font_size=22, color=OTHER_AMBER).to_edge(UP, buff=0.2)
        self.play(FadeIn(era_old))

        # Left panel: word_a neighbors
        center_a = np.array([-3.5, 0.5, 0])
        dot_a = Dot(center_a, radius=0.12, color=color_a)
        dot_a.set_glow_factor(0.4)
        dot_a_label = Text(self.word_a, font_size=14, color=color_a).next_to(dot_a, DOWN, buff=0.12)

        old_nbrs_a = get_old_neighbors(self.word_a)
        nbr_dots_a = VGroup()
        nbr_lines_a = VGroup()
        angles_a = np.linspace(0, 2 * np.pi, len(old_nbrs_a), endpoint=False)

        for i, (w, s) in enumerate(old_nbrs_a):
            offset = np.array([np.cos(angles_a[i]) * 1.3, np.sin(angles_a[i]) * 1.3, 0])
            pos = center_a + offset
            ndot = Dot(pos, radius=0.05, color=GRID_DIM).set_opacity(0.6)
            direction = offset / max(np.linalg.norm(offset), 1)
            nlabel = Text(w, font_size=11, color=SOFT_WHITE).next_to(ndot, direction, buff=0.05).set_opacity(0.8)
            nline = Line(center_a, pos, color=color_a, stroke_width=0.8).set_opacity(0.3)
            nbr_dots_a.add(ndot, nlabel)
            nbr_lines_a.add(nline)

        # Right panel: word_b neighbors
        center_b = np.array([3.5, 0.5, 0])
        dot_b = Dot(center_b, radius=0.12, color=color_b)
        dot_b.set_glow_factor(0.4)
        dot_b_label = Text(self.word_b, font_size=14, color=color_b).next_to(dot_b, DOWN, buff=0.12)

        old_nbrs_b = get_old_neighbors(self.word_b)
        nbr_dots_b = VGroup()
        nbr_lines_b = VGroup()
        angles_b = np.linspace(0, 2 * np.pi, len(old_nbrs_b), endpoint=False)

        for i, (w, s) in enumerate(old_nbrs_b):
            offset = np.array([np.cos(angles_b[i]) * 1.3, np.sin(angles_b[i]) * 1.3, 0])
            pos = center_b + offset
            ndot = Dot(pos, radius=0.05, color=GRID_DIM).set_opacity(0.6)
            direction = offset / max(np.linalg.norm(offset), 1)
            nlabel = Text(w, font_size=11, color=SOFT_WHITE).next_to(ndot, direction, buff=0.05).set_opacity(0.8)
            nline = Line(center_b, pos, color=color_b, stroke_width=0.8).set_opacity(0.3)
            nbr_dots_b.add(ndot, nlabel)
            nbr_lines_b.add(nline)

        self.play(
            FadeIn(dot_a, scale=0.5), FadeIn(dot_a_label),
            FadeIn(dot_b, scale=0.5), FadeIn(dot_b_label),
        )
        self.play(
            FadeIn(nbr_dots_a), Create(nbr_lines_a),
            FadeIn(nbr_dots_b), Create(nbr_lines_b),
        )

        old_nbr_text_a = ", ".join(w for w, _ in old_nbrs_a[:5])
        old_nbr_text_b = ", ".join(w for w, _ in old_nbrs_b[:5])
        narration_old = Text(
            f'In the 1800s: "{self.word_a}" → {old_nbr_text_a}',
            font_size=14, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.8)

        self.play(FadeIn(narration_old))
        self.wait(2.5)

        # ── Shot 4: Transition to 2000s ──
        era_new = Text("2000s", font_size=22, color=LATIN_RED).to_edge(UP, buff=0.2)

        # Get new neighbors
        new_nbrs_a = model_new.wv.most_similar(self.word_a, topn=5) if self.word_a in model_new.wv else []
        new_nbrs_b = model_new.wv.most_similar(self.word_b, topn=5) if self.word_b in model_new.wv else []

        # Ghost old neighbors
        self.play(
            ReplacementTransform(era_old, era_new),
            FadeOut(narration_old),
            nbr_dots_a.animate.set_opacity(0.15),
            nbr_lines_a.animate.set_opacity(0.1),
            nbr_dots_b.animate.set_opacity(0.15),
            nbr_lines_b.animate.set_opacity(0.1),
        )

        # New neighbor constellations (offset slightly to show movement)
        new_nbr_dots_a = VGroup()
        new_nbr_lines_a = VGroup()
        new_center_a = center_a + np.array([0.3, -0.5, 0])

        for i, (w, s) in enumerate(new_nbrs_a):
            offset = np.array([np.cos(angles_a[i] + 0.3) * 1.3, np.sin(angles_a[i] + 0.3) * 1.3, 0])
            pos = new_center_a + offset
            ndot = Dot(pos, radius=0.05, color=color_a).set_opacity(0.8)
            nlabel = Text(w, font_size=11, color=SOFT_WHITE).next_to(ndot, UR, buff=0.05).set_opacity(0.9)
            nline = Line(new_center_a, pos, color=color_a, stroke_width=0.8).set_opacity(0.4)
            new_nbr_dots_a.add(ndot, nlabel)
            new_nbr_lines_a.add(nline)

        new_nbr_dots_b = VGroup()
        new_nbr_lines_b = VGroup()
        new_center_b = center_b + np.array([-0.1, -0.1, 0])

        for i, (w, s) in enumerate(new_nbrs_b):
            offset = np.array([np.cos(angles_b[i] + 0.1) * 1.3, np.sin(angles_b[i] + 0.1) * 1.3, 0])
            pos = new_center_b + offset
            ndot = Dot(pos, radius=0.05, color=color_b).set_opacity(0.8)
            nlabel = Text(w, font_size=11, color=SOFT_WHITE).next_to(ndot, UR, buff=0.05).set_opacity(0.9)
            nline = Line(new_center_b, pos, color=color_b, stroke_width=0.8).set_opacity(0.4)
            new_nbr_dots_b.add(ndot, nlabel)
            new_nbr_lines_b.add(nline)

        self.play(
            dot_a.animate.move_to(new_center_a),
            dot_a_label.animate.next_to(Dot(new_center_a), DOWN, buff=0.12),
            dot_b.animate.move_to(new_center_b),
            dot_b_label.animate.next_to(Dot(new_center_b), DOWN, buff=0.12),
            FadeIn(new_nbr_dots_a), Create(new_nbr_lines_a),
            FadeIn(new_nbr_dots_b), Create(new_nbr_lines_b),
            run_time=2,
        )

        self.wait(1)

        # ── Shot 5: Drift score counters ──
        drift_a = info_a["drift"]
        drift_b = info_b["drift"]

        drift_color_a = LATIN_RED if drift_a > 0.5 else GREEK_GREEN
        drift_color_b = LATIN_RED if drift_b > 0.5 else GREEK_GREEN

        drift_text_a = Text(
            f"Drift: {drift_a:.3f}", font_size=22, color=drift_color_a,
        ).move_to([-3.5, -2.5, 0])

        drift_text_b = Text(
            f"Drift: {drift_b:.3f}", font_size=22, color=drift_color_b,
        ).move_to([3.5, -2.5, 0])

        self.play(
            FadeIn(drift_text_a),
            FadeIn(drift_text_b),
        )

        # Verdict
        if drift_a > drift_b:
            verdict = f'"{self.word_a}" transformed. "{self.word_b}" endured.'
        else:
            verdict = f'"{self.word_b}" transformed. "{self.word_a}" endured.'

        verdict_text = Text(
            verdict, font_size=20, color=ACCENT_PURPLE,
        ).to_edge(DOWN, buff=0.3)

        self.play(FadeIn(verdict_text, shift=UP * 0.2))
        self.wait(3)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1.5,
        )
