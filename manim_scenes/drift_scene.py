"""
Scene 2: Semantic Drift in Action (Global Manifold Visualization)
==================================================================
Cinematic 3Blue1Brown-quality overview of the entire 321-word embedding
manifold moving from 1800s to 2000s positions in UMAP subspace,
featuring origin density clustering, streamline flows, and landmark callouts.
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
    "Germanic": "#4FC3F7",   # Ice blue
    "Latin": "#EF5350",      # Crimson
    "Greek": "#66BB6A",      # Emerald
    "Sanskrit": "#FFD54F",   # Solar gold
    "PIE": "#CE93D8",        # Royal lavender
    "Other": "#90A4AE",      # Slate gray
}


class GlobalManifoldHUD(m.VGroup):
    """Top telemetry banner for the macro embedding manifold."""
    def __init__(self, n_words: int, **kwargs):
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
        title = m.Text("GLOBAL DIACHRONIC MANIFOLD", font_size=24, color="#FFFFFF")
        title.align_to(self.bg, m.LEFT).shift(m.RIGHT * 0.40)

        # Subtitle badge
        count_badge = m.Text("321 WORDS", font_size=15, color="#58A6FF")
        count_badge.next_to(title, m.RIGHT, buff=0.25)

        # Origin pill legends
        pills = []
        for origin, color in [("Germanic", "#4FC3F7"), ("Latin", "#EF5350"), ("Greek", "#66BB6A"), ("Other", "#90A4AE")]:
            dot = m.Dot(radius=0.08, color=color)
            txt = m.Text(origin, font_size=17, color=color)
            pill = m.VGroup(dot, txt).arrange(m.RIGHT, buff=0.08)
            pills.append(pill)

        pills_group = m.VGroup(*pills).arrange(m.RIGHT, buff=0.20)
        pills_group.align_to(self.bg, m.RIGHT).shift(m.LEFT * 0.35)

        self.add(title, count_badge, pills_group)
        self.move_to(m.UP * 3.20)


class LandmarkCallout(m.VGroup):
    """Clean glowing callout badge for notable drifters or anchors."""
    def __init__(self, word: str, pos: np.ndarray, score: float, category: str, color: str, offset: np.ndarray = m.UP * 0.45, **kwargs):
        super().__init__(**kwargs)
        self.ring = m.Circle(
            radius=0.26,
            stroke_color=color,
            stroke_width=2.2,
            fill_opacity=0.0,
        ).move_to(pos)
        self.dot = m.Dot(point=pos, radius=0.12, color=color)

        self.label = m.Text(
            f"{word.upper()}  ·  {category} (W2 = {score:.2f})",
            font_size=15,
            color=color,
        )
        self.label.move_to(pos + offset)
        self.label.add_background_rectangle(color="#0B0E14", opacity=0.95, buff=0.10)

        self.add(self.ring, self.dot, self.label)


class DriftScene(m.Scene):
    def construct(self) -> None:
        self.camera.background_color = DEEP_BG

        # ── 1. Load UMAP coordinates ──
        data_path = Path(__file__).resolve().parent.parent / "results" / "umap_coords.csv"
        df = pd.read_csv(data_path)

        x_vals = np.concatenate([df["x_old"].values, df["x_new"].values])
        y_vals = np.concatenate([df["y_old"].values, df["y_new"].values])

        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        x_span = max(x_max - x_min, 1e-4)
        y_span = max(y_max - y_min, 1e-4)

        # Map to viewport: x in [-5.5, 5.5], y in [-2.2, 1.8]
        target_w = 11.0
        target_h = 4.0
        scale = min(target_w / x_span, target_h / y_span) * 0.90
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        y_offset = -0.20

        def to_screen(x, y):
            return np.array([
                (x - x_center) * scale,
                (y - y_center) * scale + y_offset,
                0.0,
            ])

        # ══════════════════════════════════════════════════════
        # ACT 0 — Global Telemetry HUD & Atmosphere
        # ══════════════════════════════════════════════════════
        hud = GlobalManifoldHUD(n_words=len(df))
        era_badge = EraBadge(
            era_text="ERA: 1800 - 1900 (HISTORICAL EMBEDDING TOPOLOGY)",
            color="#58A6FF",
        )

        self.play(
            m.FadeIn(hud, shift=m.DOWN * 0.2),
            m.FadeIn(era_badge, shift=m.UP * 0.2),
            run_time=0.8,
        )

        # ══════════════════════════════════════════════════════
        # ACT I — Historical Constellation Manifestation
        # ══════════════════════════════════════════════════════
        dots_dict = {}
        old_dots_list = []

        origin_groups = []
        for origin in sorted(df["origin_class"].unique()):
            color = ORIGIN_COLORS.get(origin, "#90A4AE")
            subset = df[df["origin_class"] == origin]
            group_dots = []
            for _, row in subset.iterrows():
                pos = to_screen(row["x_old"], row["y_old"])
                dot = m.Dot(point=pos, radius=0.07, color=color)
                dots_dict[row["word"]] = dot
                old_dots_list.append(dot)
                group_dots.append(dot)
            origin_groups.append(m.VGroup(*group_dots))

        self.play(
            *[m.FadeIn(grp, scale=0.85) for grp in origin_groups],
            run_time=1.4,
        )
        self.wait(1.0)

        # ══════════════════════════════════════════════════════
        # ACT II — Global Vector Drift Flight (1800s -> 2000s)
        # ══════════════════════════════════════════════════════
        trans_era_badge = EraBadge(
            era_text="GLOBAL OPTIMAL TRANSPORT: 1800s -> 2000s MANIFOLD FLOW",
            color="#BA68C8",
        )

        flow_lines = []
        move_anims = []
        for _, row in df.iterrows():
            word = row["word"]
            p_old = to_screen(row["x_old"], row["y_old"])
            p_new = to_screen(row["x_new"], row["y_new"])
            color = ORIGIN_COLORS.get(row["origin_class"], "#90A4AE")

            flow = m.Line(
                start=p_old,
                end=p_new,
                stroke_color=color,
                stroke_opacity=0.25,
                stroke_width=1.0,
            )
            flow_lines.append(flow)
            move_anims.append(dots_dict[word].animate.move_to(p_new))

        flow_grp = m.VGroup(*flow_lines)

        self.play(
            m.Transform(era_badge, trans_era_badge),
            m.Create(flow_grp),
            *move_anims,
            run_time=2.2,
        )
        self.wait(1.0)

        # ══════════════════════════════════════════════════════
        # ACT III — Landmark Spotlights (Sequential Extremes)
        # ══════════════════════════════════════════════════════
        mod_era_badge = EraBadge(
            era_text="ERA: 2000 - 2020 (MODERN EMBEDDING TOPOLOGY)",
            color="#66BB6A",
        )

        self.play(
            m.Transform(era_badge, mod_era_badge),
            m.FadeOut(flow_grp),
            run_time=0.8,
        )

        # Spotlight 1: Top Drifter
        top_row = df.nlargest(1, "drift_score").iloc[0]
        top_pos = to_screen(top_row["x_new"], top_row["y_new"])
        top_callout = LandmarkCallout(
            word=top_row["word"],
            pos=top_pos,
            score=float(top_row["drift_score"]),
            category="MAXIMUM DRIFT",
            color="#4DD0E1",
            offset=m.UP * 0.45,
        )

        self.play(m.FadeIn(top_callout), run_time=0.8)
        self.bring_to_front(top_callout)
        self.play(m.Indicate(top_callout.ring, color="#4DD0E1", scale_factor=1.4), run_time=1.0)
        self.wait(1.0)
        self.play(m.FadeOut(top_callout), run_time=0.6)

        # Spotlight 2: Maximum Anchor / Core Invariant
        anchor_row = df.nsmallest(1, "drift_score").iloc[0]
        anchor_pos = to_screen(anchor_row["x_new"], anchor_row["y_new"])
        anchor_callout = LandmarkCallout(
            word=anchor_row["word"],
            pos=anchor_pos,
            score=float(anchor_row["drift_score"]),
            category="CORE INVARIANT",
            color="#FFD54F",
            offset=m.UP * 0.45,
        )

        self.play(m.FadeIn(anchor_callout), run_time=0.8)
        self.bring_to_front(anchor_callout)
        self.play(m.Indicate(anchor_callout.ring, color="#FFD54F", scale_factor=1.4), run_time=1.0)
        self.wait(1.2)


if __name__ == "__main__":
    print("DriftScene configured for global manifold visualization.")
