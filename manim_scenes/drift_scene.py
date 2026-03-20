"""
Scene 2: "Semantic Drift in Action" (Main Drift Visualization)
================================================================
The centerpiece: shows tracked words moving through UMAP embedding
space from 1800s to 2000s positions. Data-driven from umap_coords.csv.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Colors ──
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
    "Latin":    LATIN_RED,
    "Greek":    GREEK_GREEN,
    "Other":    OTHER_AMBER,
}


class DriftScene(Scene):
    def construct(self):
        self.camera.background_color = DEEP_BG

        # ── Load data ──
        data_path = Path(__file__).resolve().parent.parent / "results" / "umap_coords.csv"
        df = pd.read_csv(data_path)

        # Normalize UMAP coords to fit Manim's coordinate system
        x_vals = np.concatenate([df["x_old"].values, df["x_new"].values])
        y_vals = np.concatenate([df["y_old"].values, df["y_new"].values])

        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        scale = min(10 / x_range, 6.5 / y_range) * 0.85
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        def to_manim(x, y):
            return np.array([
                (x - x_center) * scale,
                (y - y_center) * scale,
                0,
            ])

        # ── Shot 1: Title ──
        title = Text(
            "Semantic Drift: 1800s → 2000s",
            font_size=42, color=SOFT_WHITE,
        )
        subtitle = Text(
            f"{len(df)} words across 4 etymological origins",
            font_size=22, color=ACCENT_PURPLE,
        ).next_to(title, DOWN, buff=0.4)

        self.play(Write(title, run_time=1.5))
        self.play(FadeIn(subtitle, shift=UP * 0.2))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ── Shot 2: Axes area + UMAP label ──
        umap_label = Text(
            "UMAP projection of 100-dimensional word embeddings",
            font_size=16, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.3).set_opacity(0.5)
        self.play(FadeIn(umap_label))

        # ── Shot 3: Legend ──
        legend_items = VGroup()
        for i, (origin, color) in enumerate(ORIGIN_COLORS.items()):
            dot = Dot(radius=0.08, color=color)
            label = Text(origin, font_size=16, color=color)
            item = VGroup(dot, label).arrange(RIGHT, buff=0.15)
            legend_items.add(item)

        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend_box = SurroundingRectangle(
            legend_items, buff=0.2, color=GRID_DIM, fill_color=DEEP_BG,
            fill_opacity=0.8, corner_radius=0.1,
        )
        legend = VGroup(legend_box, legend_items).to_corner(UL, buff=0.3)
        self.play(FadeIn(legend))

        # ── Shot 4: Words appear at 1800s positions ──
        narration1 = Text(
            "Each dot is a word, positioned by its meaning in the 1800s",
            font_size=20, color=SOFT_WHITE,
        ).to_edge(UP, buff=0.3)

        self.play(FadeIn(narration1, shift=DOWN * 0.2))

        dots = {}
        labels_group = {}
        # Sort by origin for grouped appearance
        for origin in ["Germanic", "Latin", "Greek", "Other"]:
            origin_df = df[df["origin_class"] == origin]
            origin_group = VGroup()
            for _, row in origin_df.iterrows():
                pos = to_manim(row["x_old"], row["y_old"])
                color = ORIGIN_COLORS[origin]

                dot = Dot(pos, radius=0.06, color=color)
                dot.set_glow_factor(0.2)
                label = Text(
                    row["word"], font_size=9, color=color,
                ).next_to(dot, UR, buff=0.05).set_opacity(0.7)

                dots[row["word"]] = dot
                labels_group[row["word"]] = label
                origin_group.add(dot, label)

            # Staggered fade in per origin group
            self.play(
                LaggedStart(
                    *[FadeIn(mob, scale=0.5) for mob in origin_group],
                    lag_ratio=0.05,
                ),
                run_time=1.2,
            )

        self.wait(1.5)

        # ── Shot 5: Epoch label ──
        self.play(FadeOut(narration1))
        epoch_label = Text(
            "1800–1900", font_size=36, color=OTHER_AMBER,
        ).to_edge(UP, buff=0.3)
        self.play(Write(epoch_label))
        self.wait(1.5)

        # ── Shot 6 & 7: Animate transition to 2000s ──
        epoch_new = Text(
            "2000–2020", font_size=36, color=LATIN_RED,
        ).to_edge(UP, buff=0.3)

        narration2 = Text(
            "Watch them move to where they live in modern language...",
            font_size=20, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.6)

        self.play(
            ReplacementTransform(epoch_label, epoch_new),
            FadeIn(narration2),
        )

        # Create trail lines and animate movement
        trails = VGroup()
        animations = []

        for _, row in df.iterrows():
            word = row["word"]
            old_pos = to_manim(row["x_old"], row["y_old"])
            new_pos = to_manim(row["x_new"], row["y_new"])
            color = ORIGIN_COLORS[row["origin_class"]]

            # Trail line
            trail = Line(old_pos, new_pos, color=color, stroke_width=1).set_opacity(0.25)
            trails.add(trail)

            # Animate dot movement
            animations.append(dots[word].animate.move_to(new_pos))
            # Animate label movement
            target_label_pos = new_pos + np.array([0.1, 0.1, 0])
            animations.append(labels_group[word].animate.move_to(target_label_pos))

        self.play(
            Create(trails, run_time=0.5),
            *animations,
            run_time=4,
        )

        self.wait(1)
        self.play(FadeOut(narration2))

        # ── Shot 8: Highlight top drifters ──
        top_drift = df.nlargest(3, "drift_score")
        highlights = VGroup()

        narration3 = Text(
            "Some words barely moved. Others traveled enormously.",
            font_size=20, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(narration3))

        for _, row in top_drift.iterrows():
            word = row["word"]
            dot = dots[word]

            # Pulsing glow
            glow = Circle(radius=0.3, color=ACCENT_PURPLE).move_to(dot.get_center())
            glow.set_fill(ACCENT_PURPLE, opacity=0.1)
            glow.set_stroke(ACCENT_PURPLE, width=2, opacity=0.6)

            score_text = Text(
                f"{word}: {row['drift_score']:.2f}",
                font_size=18, color=ACCENT_PURPLE,
            ).next_to(glow, DOWN, buff=0.15)

            highlights.add(glow, score_text)
            self.play(
                Create(glow),
                FadeIn(score_text),
                run_time=0.7,
            )
            self.play(
                glow.animate.scale(1.2).set_opacity(0.3),
                rate_func=there_and_back,
                run_time=0.5,
            )

        self.wait(2)

        # ── Shot 9: Most stable words ──
        self.play(FadeOut(highlights), FadeOut(narration3))

        bottom_drift = df.nsmallest(3, "drift_score")
        stable_group = VGroup()

        narration4 = Text(
            "These words barely changed — their meaning endured.",
            font_size=20, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(narration4))

        for _, row in bottom_drift.iterrows():
            word = row["word"]
            dot = dots[word]

            glow = Circle(radius=0.25, color=GREEK_GREEN).move_to(dot.get_center())
            glow.set_fill(GREEK_GREEN, opacity=0.1)
            glow.set_stroke(GREEK_GREEN, width=2, opacity=0.6)

            score_text = Text(
                f"{word}: {row['drift_score']:.2f}",
                font_size=18, color=GREEK_GREEN,
            ).next_to(glow, DOWN, buff=0.15)

            stable_group.add(glow, score_text)
            self.play(
                Create(glow), FadeIn(score_text),
                run_time=0.5,
            )

        self.wait(2.5)

        # ── Outro ──
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1.5,
        )
