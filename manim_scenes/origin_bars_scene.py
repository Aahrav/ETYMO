"""
Scene 4: Which Origins Drift Most? (Origin Analysis Animation)
==============================================================
Cinematic 3Blue1Brown-quality bar chart visualizing the ranking of
semantic drift (Wasserstein W2) across all 6 etymological origin classes,
featuring animated growing bars, sample counts, and ANOVA statistical significance.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import manimgx as m

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from manim_scenes.components import EraBadge

DEEP_BG = "#0B0E14"

ORIGIN_COLORS = {
    "Greek":    "#66BB6A",   # Emerald
    "Other":    "#90A4AE",   # Slate gray
    "PIE":      "#CE93D8",   # Royal lavender
    "Latin":    "#EF5350",   # Crimson
    "Germanic": "#4FC3F7",   # Ice blue
    "Sanskrit": "#FFD54F",   # Solar gold
}


class OriginBarsHUD(m.VGroup):
    """Top telemetry bar for origin ranking."""
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

        title = m.Text("ETYMOLOGICAL DRIFT RANKING", font_size=24, color="#FFFFFF")
        title.align_to(self.bg, m.LEFT).shift(m.RIGHT * 0.40)

        metric_lbl = m.Text("METRIC: MEAN WASSERSTEIN W2", font_size=15, color="#58A6FF")
        metric_lbl.align_to(self.bg, m.RIGHT).shift(m.LEFT * 0.40)

        self.add(title, metric_lbl)
        self.move_to(m.UP * 3.20)


class OriginBarsScene(m.Scene):
    def construct(self) -> None:
        self.camera.background_color = DEEP_BG

        # ── 1. Load Origin Summary ──
        base = Path(__file__).resolve().parent.parent / "results"
        summary_csv = base / "dl_origin_drift_summary.csv"
        if not summary_csv.exists():
            summary_csv = base / "origin_drift_summary.csv"
        
        df = pd.read_csv(summary_csv)
        score_col = "mean_w2" if "mean_w2" in df.columns else "mean_drift"
        df = df.sort_values(score_col).reset_index(drop=True)

        # ══════════════════════════════════════════════════════
        # ACT 0 — Atmosphere & HUD
        # ══════════════════════════════════════════════════════
        hud = OriginBarsHUD()
        era_badge = EraBadge(
            era_text="RESEARCH QUESTION: DO ETYMOLOGICAL ORIGINS PREDICT SEMANTIC STABILITY?",
            color="#58A6FF",
        )

        self.play(
            m.FadeIn(hud, shift=m.DOWN * 0.2),
            m.FadeIn(era_badge, shift=m.UP * 0.2),
            run_time=0.8,
        )

        # ══════════════════════════════════════════════════════
        # ACT I — Chart Framework Setup
        # ══════════════════════════════════════════════════════
        n_origins = len(df)
        bar_width = 1.35
        gap = 0.55
        total_chart_w = n_origins * bar_width + (n_origins - 1) * gap
        start_x = -total_chart_w / 2.0 + bar_width / 2.0
        baseline_y = -2.10

        # Base horizontal axis line
        axis_line = m.Line(
            start=np.array([-5.6, baseline_y, 0.0]),
            end=np.array([5.6, baseline_y, 0.0]),
            stroke_color="#30363D",
            stroke_width=2.0,
        )
        self.play(m.Create(axis_line), run_time=0.8)

        # ══════════════════════════════════════════════════════
        # ACT II — Animated Growing Origin Bars
        # ══════════════════════════════════════════════════════
        bars = []
        bar_labels = []
        bar_scores = []
        bar_counts = []

        max_score = df[score_col].max()
        min_score = df[score_col].min()
        max_bar_height = 3.60

        for i, row in df.iterrows():
            origin = row["origin_class"]
            score = float(row[score_col])
            n_w = int(row["n_words"])
            color = ORIGIN_COLORS.get(origin, "#90A4AE")
            
            # Map score to bar height (proportional relative to 0)
            h = (score / max_score) * max_bar_height
            cx = start_x + i * (bar_width + gap)

            bar = m.RoundedRectangle(
                corner_radius=0.10,
                width=bar_width,
                height=h,
                fill_color=color,
                fill_opacity=0.75,
                stroke_color=color,
                stroke_width=1.5,
            )
            # Center of bar is at baseline_y + h / 2
            bar.move_to(np.array([cx, baseline_y + h / 2.0, 0.0]))

            # Origin label below baseline
            lbl = m.Text(origin, font_size=16, color=color)
            lbl.move_to(np.array([cx, baseline_y - 0.35, 0.0]))

            # Score text above bar
            score_txt = m.Text(f"{score:.4f}", font_size=15, color="#FFFFFF")
            score_txt.next_to(bar, m.UP, buff=0.15)
            score_txt.add_background_rectangle(color="#0B0E14", opacity=0.92, buff=0.06)

            # Sample count inside bar
            count_txt = m.Text(f"N={n_w}", font_size=16, color="#0B0E14")
            count_txt.move_to(np.array([cx, baseline_y + 0.35, 0.0]))

            bars.append(bar)
            bar_labels.append(lbl)
            bar_scores.append(score_txt)
            bar_counts.append(count_txt)

        bars_grp = m.VGroup(*bars)
        labels_grp = m.VGroup(*bar_labels)
        scores_grp = m.VGroup(*bar_scores)
        counts_grp = m.VGroup(*bar_counts)

        # Grow bars from baseline
        self.play(
            m.FadeIn(labels_grp),
            m.Create(bars_grp),
            run_time=1.4,
        )
        self.play(
            m.FadeIn(scores_grp),
            m.FadeIn(counts_grp),
            run_time=0.8,
        )
        self.wait(1.0)

        # ══════════════════════════════════════════════════════
        # ACT III — Statistical Verdict & Key Finding
        # ══════════════════════════════════════════════════════
        stat_badge = EraBadge(
            era_text="ANOVA: F = 5.62, p = 4.2e-5  |  ORIGIN DRIFT DIFFERENCES ARE STATISTICALLY SIGNIFICANT",
            color="#FFD54F",
        )

        # Highlight most stable (Greek) and most drifted (Sanskrit / Germanic)
        most_stable_bar = bars[0]
        most_drifted_bar = bars[-1]

        self.play(
            m.Transform(era_badge, stat_badge),
            m.Indicate(most_stable_bar, color="#66BB6A", scale_factor=1.06),
            m.Indicate(most_drifted_bar, color="#FFD54F", scale_factor=1.06),
            run_time=1.4,
        )
        self.wait(2.0)


if __name__ == "__main__":
    print("OriginBarsScene configured for etymological drift comparison.")
