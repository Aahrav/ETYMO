"""
Scene 4: "Which Origins Drift Most?" (Origin Analysis Animation)
==================================================================
Builds the key finding — origin-wise drift comparison — as a visual
narrative with animated growing bars, beeswarm overlay, and stats.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


class OriginBarsScene(Scene):
    def construct(self):
        self.camera.background_color = DEEP_BG

        # ── Load data ──
        base = Path(__file__).resolve().parent.parent / "results"
        origin_df = pd.read_csv(base / "origin_drift_summary.csv")
        drift_df = pd.read_csv(base / "drift_scores.csv")
        drift_df = drift_df[drift_df["status"] == "OK"]

        # Sort by mean drift
        origin_df = origin_df.sort_values("mean_drift").reset_index(drop=True)

        # ── Shot 1: Title ──
        title = Text(
            "Do etymological origins\npredict semantic stability?",
            font_size=42, color=SOFT_WHITE,
        )
        subtitle = Text(
            "This is the core question.",
            font_size=22, color=ACCENT_PURPLE,
        ).next_to(title, DOWN, buff=0.4)

        self.play(Write(title, run_time=1.5))
        self.play(FadeIn(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ── Shot 2: Empty axes ──
        chart_left = -4.5
        chart_right = 4.5
        chart_bottom = -2.5
        chart_top = 2.8
        chart_width = chart_right - chart_left

        # Y-axis
        y_axis = Line(
            [chart_left, chart_bottom, 0],
            [chart_left, chart_top, 0],
            color=SOFT_WHITE, stroke_width=1.5,
        )
        y_label = Text(
            "Mean Semantic Drift", font_size=16, color=SOFT_WHITE,
        ).rotate(PI / 2).next_to(y_axis, LEFT, buff=0.3)

        # X-axis
        x_axis = Line(
            [chart_left, chart_bottom, 0],
            [chart_right, chart_bottom, 0],
            color=SOFT_WHITE, stroke_width=1.5,
        )

        # Y-axis ticks
        y_ticks = VGroup()
        max_val = 0.9
        for val in np.arange(0, max_val + 0.1, 0.2):
            y = chart_bottom + (val / max_val) * (chart_top - chart_bottom)
            tick = Line([chart_left - 0.1, y, 0], [chart_left, y, 0],
                       color=SOFT_WHITE, stroke_width=1)
            tick_label = Text(f"{val:.1f}", font_size=12, color=SOFT_WHITE).next_to(tick, LEFT, buff=0.1)
            y_ticks.add(tick, tick_label)
            # Grid line
            grid_line = DashedLine(
                [chart_left, y, 0], [chart_right, y, 0],
                color=GRID_DIM, stroke_width=0.5,
            )
            y_ticks.add(grid_line)

        self.play(Create(y_axis), Create(x_axis), FadeIn(y_label))
        self.play(FadeIn(y_ticks))

        # X-axis labels
        n_bars = len(origin_df)
        bar_width = chart_width / (n_bars * 1.8)
        bar_positions = []

        x_labels = VGroup()
        for i, (_, row) in enumerate(origin_df.iterrows()):
            x_pos = chart_left + (i + 0.5) * (chart_width / n_bars)
            bar_positions.append(x_pos)
            label = Text(
                row["origin_class"], font_size=16,
                color=ORIGIN_COLORS[row["origin_class"]],
            ).move_to([x_pos, chart_bottom - 0.3, 0])
            x_labels.add(label)

        self.play(FadeIn(x_labels))
        self.wait(0.5)

        # ── Shot 3: Bars grow one at a time ──
        bars = VGroup()
        value_labels = VGroup()

        for i, (_, row) in enumerate(origin_df.iterrows()):
            origin = row["origin_class"]
            mean_drift = row["mean_drift"]
            color = ORIGIN_COLORS[origin]

            bar_height = (mean_drift / max_val) * (chart_top - chart_bottom)
            x_pos = bar_positions[i]

            bar = Rectangle(
                width=bar_width,
                height=bar_height,
                fill_color=color,
                fill_opacity=0.8,
                stroke_color=SOFT_WHITE,
                stroke_width=0.5,
            )
            bar.move_to([x_pos, chart_bottom + bar_height / 2, 0])

            # Value counter that animates from 0
            val_text = Text(
                f"{mean_drift:.3f}", font_size=16, color=SOFT_WHITE,
            ).next_to(bar, UP, buff=0.15)

            narration = Text(
                f"{origin} words drifted an average of {mean_drift:.3f}",
                font_size=18, color=color,
            ).to_edge(DOWN, buff=0.5)

            # Grow bar from bottom
            bar_copy = bar.copy()
            bar_copy.stretch_to_fit_height(0.01)
            bar_copy.move_to([x_pos, chart_bottom, 0], aligned_edge=DOWN)

            self.play(
                Transform(bar_copy, bar),
                FadeIn(narration),
                run_time=1.0,
                rate_func=rate_functions.ease_out_bounce,
            )
            self.play(FadeIn(val_text), run_time=0.3)
            bars.add(bar_copy)
            value_labels.add(val_text)

            if i < n_bars - 1:
                self.play(FadeOut(narration), run_time=0.3)
            else:
                self.wait(1)
                self.play(FadeOut(narration))

        self.wait(0.5)

        # ── Shot 4: Error bars ──
        narration_var = Text(
            "But there's variance within each group...",
            font_size=18, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(narration_var))

        error_bars = VGroup()
        for i, (_, row) in enumerate(origin_df.iterrows()):
            mean = row["mean_drift"]
            std = row["std_drift"]
            x_pos = bar_positions[i]

            y_mean = chart_bottom + (mean / max_val) * (chart_top - chart_bottom)
            y_top = chart_bottom + (min(mean + std, max_val) / max_val) * (chart_top - chart_bottom)
            y_bot = chart_bottom + (max(mean - std, 0) / max_val) * (chart_top - chart_bottom)

            whisker = VGroup(
                Line([x_pos, y_bot, 0], [x_pos, y_top, 0], color=SOFT_WHITE, stroke_width=2),
                Line([x_pos - 0.1, y_top, 0], [x_pos + 0.1, y_top, 0], color=SOFT_WHITE, stroke_width=2),
                Line([x_pos - 0.1, y_bot, 0], [x_pos + 0.1, y_bot, 0], color=SOFT_WHITE, stroke_width=2),
            )
            error_bars.add(whisker)

        self.play(Create(error_bars, run_time=1.0))
        self.wait(1)
        self.play(FadeOut(narration_var))

        # ── Shot 5 & 6: Highlight extremes ──
        most_stable_idx = 0
        most_drifted_idx = n_bars - 1
        stable_origin = origin_df.iloc[most_stable_idx]["origin_class"]
        drifted_origin = origin_df.iloc[most_drifted_idx]["origin_class"]

        # Highlight most stable
        stable_glow = SurroundingRectangle(
            bars[most_stable_idx],
            color=GREEK_GREEN, buff=0.1, corner_radius=0.05,
        ).set_stroke(width=3)

        stable_text = Text(
            f"{stable_origin} words were the most stable over 200 years",
            font_size=18, color=GREEK_GREEN,
        ).to_edge(DOWN, buff=0.5)

        self.play(Create(stable_glow), FadeIn(stable_text))
        self.play(
            stable_glow.animate.scale(1.05),
            rate_func=there_and_back, run_time=0.6,
        )
        self.wait(1.5)
        self.play(FadeOut(stable_glow), FadeOut(stable_text))

        # Highlight most drifted
        drift_glow = SurroundingRectangle(
            bars[most_drifted_idx],
            color=LATIN_RED, buff=0.1, corner_radius=0.05,
        ).set_stroke(width=3)

        drift_text = Text(
            f"{drifted_origin} words showed the most semantic change",
            font_size=18, color=LATIN_RED,
        ).to_edge(DOWN, buff=0.5)

        self.play(Create(drift_glow), FadeIn(drift_text))
        self.play(
            drift_glow.animate.scale(1.05),
            rate_func=there_and_back, run_time=0.6,
        )
        self.wait(1.5)
        self.play(FadeOut(drift_glow), FadeOut(drift_text))

        # ── Shot 7: Beeswarm overlay ──
        narration_bee = Text(
            "Each dot is one of our tracked words",
            font_size=18, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(narration_bee))

        np.random.seed(42)
        word_dots = VGroup()
        for i, (_, row) in enumerate(origin_df.iterrows()):
            origin = row["origin_class"]
            color = ORIGIN_COLORS[origin]
            x_pos = bar_positions[i]

            word_drifts = drift_df[drift_df["origin_class"] == origin]["drift_score"].values
            jitter = np.random.uniform(-bar_width * 0.35, bar_width * 0.35, len(word_drifts))

            for j, d in enumerate(word_drifts):
                y = chart_bottom + (d / max_val) * (chart_top - chart_bottom)
                dot = Dot(
                    [x_pos + jitter[j], y, 0],
                    radius=0.04, color=color,
                ).set_opacity(0.8)
                # Start from top and rain down
                dot_start = dot.copy().move_to([x_pos + jitter[j], chart_top + 0.5, 0])
                word_dots.add(dot)

        self.play(
            LaggedStart(
                *[FadeIn(d, shift=DOWN * 0.5) for d in word_dots],
                lag_ratio=0.02,
            ),
            run_time=2,
        )
        self.wait(1)
        self.play(FadeOut(narration_bee))

        # ── Shot 8: Statistical significance ──
        # Bracket between most stable and most drifted
        p_value = 0.061  # From Phase 8
        sig_text = Text(
            f"p = {p_value:.3f} (Kruskal-Wallis)",
            font_size=16, color=ACCENT_PURPLE,
        ).move_to([0, chart_top + 0.3, 0])

        bracket_line = Line(
            [bar_positions[0], chart_top + 0.1, 0],
            [bar_positions[-1], chart_top + 0.1, 0],
            color=ACCENT_PURPLE, stroke_width=1.5,
        )

        self.play(Create(bracket_line), FadeIn(sig_text))
        self.wait(1.5)

        # ── Shot 9: Final insight ──
        final_insight = Text(
            "Germanic words are significantly more stable\n"
            "than Greek and Other-origin words (pairwise p < 0.05)",
            font_size=18, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.4)

        self.play(FadeIn(final_insight, shift=UP * 0.2))
        self.wait(3)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1.5,
        )
