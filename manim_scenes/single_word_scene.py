"""
Scene: Single Word Drift (Explorer Deep-Dive)
================================================
Shows a single word's semantic drift — its neighbor constellation
in 1800s morphing into the 2000s constellation.

This scene is PARAMETRIC — `target_word` is set dynamically by the
web app's render pipeline.

Default word: "craft"
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
SANSKRIT_GOLD = "#FFD54F"
PIE_LAVENDER = "#CE93D8"
OTHER_GRAY = "#90A4AE"
SOFT_WHITE = "#e0e0e0"
ACCENT_PURPLE = "#BB86FC"
GRID_DIM = "#202040"

ORIGIN_COLORS = {
    "Germanic": GERMANIC_BLUE,
    "Latin": LATIN_RED,
    "Greek": GREEK_GREEN,
    "Sanskrit": SANSKRIT_GOLD,
    "PIE": PIE_LAVENDER,
    "Other": OTHER_GRAY,
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


class SingleWordDriftScene(Scene):
    target_word = "craft"

    def construct(self):
        self.camera.background_color = DEEP_BG

        model_old, model_new, aligned_old, old_words, old_w2i, drift_df = load_data()

        word = self.target_word

        # ── Word info ──
        row = drift_df[drift_df["word"] == word]
        if len(row) > 0:
            row = row.iloc[0]
            origin = row["origin_class"]
            drift = row["drift_score"] if row["status"] == "OK" else 0
        else:
            origin = "Unknown"
            drift = 0

        color = ORIGIN_COLORS.get(origin, SOFT_WHITE)

        # ── Old neighbors ──
        old_nbrs = []
        if word in old_w2i:
            target_vec = aligned_old[old_w2i[word]]
            norms = np.linalg.norm(aligned_old, axis=1)
            norms[norms == 0] = 1
            sims = aligned_old @ target_vec / (norms * np.linalg.norm(target_vec))
            top_idxs = np.argsort(sims)[::-1][1:6]
            old_nbrs = [(old_words[i], float(sims[i])) for i in top_idxs]

        # ── New neighbors ──
        new_nbrs = model_new.wv.most_similar(word, topn=5) if word in model_new.wv else []

        shared = set(w for w, _ in old_nbrs) & set(w for w, _ in new_nbrs)

        # ════════════════════════════════════════════
        # SHOT 1 — Title card
        # ════════════════════════════════════════════
        title = Text(f'"{word}"', font_size=52, color=color)
        subtitle = Text(
            f"Semantic Drift Analysis  •  {origin}",
            font_size=20, color=SOFT_WHITE,
        ).next_to(title, DOWN, buff=0.35)

        self.play(FadeIn(title, scale=0.9))
        self.play(FadeIn(subtitle))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ════════════════════════════════════════════
        # SHOT 2 — 1800s neighbor constellation
        # ════════════════════════════════════════════
        era_label = Text("1800s", font_size=26, color=SANSKRIT_GOLD).to_edge(UP, buff=0.3)
        self.play(FadeIn(era_label))

        center = np.array([0.0, 0.0, 0.0])
        main_dot = Dot(center, radius=0.18, color=color)
        main_dot.set_glow_factor(0.5)
        main_label = Text(word, font_size=20, color=color).next_to(main_dot, DOWN, buff=0.18)

        self.play(FadeIn(main_dot, scale=0.5), FadeIn(main_label))

        num_old = len(old_nbrs)
        angles = np.linspace(0, 2 * np.pi, max(num_old, 5), endpoint=False)

        old_group = VGroup()
        old_lines = VGroup()

        for i, (w, s) in enumerate(old_nbrs):
            r = 2.2
            offset = np.array([np.cos(angles[i]) * r, np.sin(angles[i]) * r, 0])
            pos = center + offset

            dot = Dot(pos, radius=0.07, color=GRID_DIM).set_opacity(0.7)
            direction = offset / max(np.linalg.norm(offset), 1)
            label = Text(w, font_size=14, color=SOFT_WHITE).next_to(
                dot, direction, buff=0.08
            ).set_opacity(0.85)
            sim_label = Text(
                f"{s:.3f}", font_size=10, color=SOFT_WHITE
            ).next_to(label, DOWN, buff=0.04).set_opacity(0.4)

            line = Line(center, pos, color=color, stroke_width=1.0).set_opacity(0.3)

            old_group.add(dot, label, sim_label)
            old_lines.add(line)

        self.play(Create(old_lines), FadeIn(old_group), run_time=1.5)

        narration = Text(
            f'Historical context: {", ".join(w for w, _ in old_nbrs[:5])}',
            font_size=14, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.5).set_opacity(0.7)
        self.play(FadeIn(narration))
        self.wait(2)

        # ════════════════════════════════════════════
        # SHOT 3 — Transition to 2000s
        # ════════════════════════════════════════════
        era_new = Text("2000s", font_size=26, color=LATIN_RED).to_edge(UP, buff=0.3)

        self.play(
            ReplacementTransform(era_label, era_new),
            FadeOut(narration),
            old_group.animate.set_opacity(0.12),
            old_lines.animate.set_opacity(0.08),
            run_time=1.0,
        )

        # Shift center slightly to show movement
        new_center = np.array([0.4, -0.3, 0])
        num_new = len(new_nbrs)
        angles_new = np.linspace(0, 2 * np.pi, max(num_new, 5), endpoint=False)

        new_group = VGroup()
        new_lines = VGroup()

        for i, (w, s) in enumerate(new_nbrs):
            r = 2.2
            angle_offset = 0.4  # rotate new constellation slightly
            offset = np.array([
                np.cos(angles_new[i] + angle_offset) * r,
                np.sin(angles_new[i] + angle_offset) * r,
                0,
            ])
            pos = new_center + offset

            is_shared = w in shared
            dot_color = ACCENT_PURPLE if is_shared else color
            dot = Dot(pos, radius=0.07, color=dot_color).set_opacity(0.85)
            direction = offset / max(np.linalg.norm(offset), 1)
            label = Text(w, font_size=14, color=SOFT_WHITE).next_to(
                dot, direction, buff=0.08
            ).set_opacity(0.9)
            sim_label = Text(
                f"{s:.3f}", font_size=10, color=SOFT_WHITE
            ).next_to(label, DOWN, buff=0.04).set_opacity(0.4)

            line = Line(new_center, pos, color=color, stroke_width=1.0).set_opacity(0.4)

            new_group.add(dot, label, sim_label)
            new_lines.add(line)

        self.play(
            main_dot.animate.move_to(new_center),
            main_label.animate.next_to(Dot(new_center), DOWN, buff=0.18),
            run_time=1.2,
        )
        self.play(Create(new_lines), FadeIn(new_group), run_time=1.5)

        narration_new = Text(
            f'Modern context: {", ".join(w for w, _ in new_nbrs[:5])}',
            font_size=14, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.5).set_opacity(0.7)
        self.play(FadeIn(narration_new))
        self.wait(2)

        # ════════════════════════════════════════════
        # SHOT 4 — Drift score reveal
        # ════════════════════════════════════════════
        self.play(FadeOut(narration_new))

        drift_color = LATIN_RED if drift > 0.5 else GREEK_GREEN
        drift_level = "High Shift" if drift > 0.5 else ("Moderate" if drift > 0.3 else "Stable")

        score_box = VGroup()
        score_title = Text("Semantic Drift Score", font_size=16, color=SOFT_WHITE).set_opacity(0.6)
        score_value = Text(f"{drift:.4f}", font_size=42, color=drift_color)
        score_level = Text(drift_level, font_size=18, color=drift_color)
        score_box.add(score_title, score_value, score_level)
        score_box.arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.6)

        shared_count = len(shared)
        shared_text = Text(
            f"Neighbors retained: {shared_count}/5",
            font_size=14, color=ACCENT_PURPLE,
        ).next_to(score_box, DOWN, buff=0.2)

        self.play(FadeIn(score_box, shift=UP * 0.3))
        self.play(FadeIn(shared_text))
        self.wait(3)

        # ── Fade out ──
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
