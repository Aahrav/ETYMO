"""
Scene 5: "The Procrustes Alignment" (Method Explainer)
========================================================
Visually explains WHY we need alignment and HOW Orthogonal Procrustes works.
This is the hardest concept for viewers — makes it intuitive.
"""

import sys
from pathlib import Path
import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEEP_BG = "#1a1a2e"
GERMANIC_BLUE = "#4FC3F7"
LATIN_RED = "#EF5350"
GREEK_GREEN = "#66BB6A"
OTHER_AMBER = "#FFA726"
SOFT_WHITE = "#e0e0e0"
ACCENT_PURPLE = "#BB86FC"
ACCENT_GOLD = "#FFD54F"
GRID_DIM = "#2a2a4e"


class AlignmentScene(Scene):
    def construct(self):
        self.camera.background_color = DEEP_BG

        # ── Shot 1: Two coordinate grids side by side ──
        title = Text(
            "We trained two separate\nembedding models...",
            font_size=36, color=SOFT_WHITE,
        )
        self.play(Write(title, run_time=1.5))
        self.wait(1.5)
        self.play(FadeOut(title))

        # Left grid (1800s space)
        left_axes = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=4, y_length=4,
            tips=False,
            axis_config={"color": GRID_DIM, "stroke_width": 1},
        ).shift(LEFT * 3.3)

        left_label = Text("1800s Space", font_size=18, color=OTHER_AMBER,
                         ).next_to(left_axes, UP, buff=0.3)

        # Right grid (2000s space)
        right_axes = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=4, y_length=4,
            tips=False,
            axis_config={"color": GRID_DIM, "stroke_width": 1},
        ).shift(RIGHT * 3.3)

        right_label = Text("2000s Space", font_size=18, color=LATIN_RED,
                          ).next_to(right_axes, UP, buff=0.3)

        self.play(
            Create(left_axes), Create(right_axes),
            FadeIn(left_label), FadeIn(right_label),
        )

        # ── Shot 2: Same words at DIFFERENT positions (mismatched spaces) ──
        # Word positions in left (1800s) space
        words_left = {
            "water": (-1, 2, GERMANIC_BLUE),
            "sun":   (-2, 1, GERMANIC_BLUE),
            "stone": (0.5, 1.5, GERMANIC_BLUE),
            "mouse": (1, -1, GERMANIC_BLUE),
            "board": (-1, -2, GERMANIC_BLUE),
        }

        # The SAME words but at different positions in right space (random rotation)
        # This illustrates that the two spaces are NOT aligned
        words_right = {
            "water": (1.5, -1.5, GERMANIC_BLUE),
            "sun":   (0.5, -2.5, GERMANIC_BLUE),
            "stone": (2, 0, GERMANIC_BLUE),
            "mouse": (-1.5, 1, GERMANIC_BLUE),
            "board": (0, 2, GERMANIC_BLUE),
        }

        left_dots = {}
        right_dots = {}

        for word, (x, y, color) in words_left.items():
            pos = left_axes.c2p(x, y)
            dot = Dot(pos, radius=0.08, color=color)
            label = Text(word, font_size=12, color=color).next_to(dot, UR, buff=0.05)
            left_dots[word] = VGroup(dot, label)

        for word, (x, y, color) in words_right.items():
            pos = right_axes.c2p(x, y)
            dot = Dot(pos, radius=0.08, color=color)
            label = Text(word, font_size=12, color=color).next_to(dot, UR, buff=0.05)
            right_dots[word] = VGroup(dot, label)

        for word in words_left:
            self.play(
                FadeIn(left_dots[word], scale=0.5),
                FadeIn(right_dots[word], scale=0.5),
                run_time=0.3,
            )

        self.wait(1)

        # ── Shot 3: Show the mismatch ──
        narration_problem = Text(
            "Problem: the two spaces are NOT aligned.",
            font_size=22, color=LATIN_RED,
        ).to_edge(DOWN, buff=0.5)

        # Draw red ✗ between corresponding pairs
        cross_marks = VGroup()
        for word in words_left:
            mid = (left_dots[word][0].get_center() + right_dots[word][0].get_center()) / 2
            cross = Text("✗", font_size=24, color=LATIN_RED).move_to(mid).set_opacity(0.7)
            cross_marks.add(cross)

        # Dashed lines showing mismatch
        mismatch_lines = VGroup()
        for word in words_left:
            line = DashedLine(
                left_dots[word][0].get_center(),
                right_dots[word][0].get_center(),
                color=LATIN_RED, stroke_width=1,
            ).set_opacity(0.3)
            mismatch_lines.add(line)

        self.play(
            Create(mismatch_lines),
            FadeIn(cross_marks),
            FadeIn(narration_problem),
        )

        narration_meaningless = Text(
            "Comparing vectors directly would be meaningless.",
            font_size=18, color=SOFT_WHITE,
        ).next_to(narration_problem, UP, buff=0.2)
        self.play(FadeIn(narration_meaningless))
        self.wait(2)
        self.play(FadeOut(narration_meaningless), FadeOut(narration_problem))
        self.play(FadeOut(cross_marks), FadeOut(mismatch_lines))

        # ── Shot 4: Highlight anchor words in gold ──
        anchor_words = ["water", "sun", "stone"]
        other_words = ["mouse", "board"]

        narration_anchor = Text(
            "Solution: use words we KNOW haven't changed as anchors.",
            font_size=20, color=ACCENT_GOLD,
        ).to_edge(DOWN, buff=0.5)

        self.play(FadeIn(narration_anchor))

        for word in anchor_words:
            # Highlight in gold
            for dot_group in [left_dots[word], right_dots[word]]:
                self.play(
                    dot_group[0].animate.set_color(ACCENT_GOLD).scale(1.3),
                    dot_group[1].animate.set_color(ACCENT_GOLD),
                    run_time=0.4,
                )

        anchor_label = Text(
            'Anchor words: "water", "sun", "stone"',
            font_size=16, color=ACCENT_GOLD,
        ).to_edge(UP, buff=0.3)
        self.play(FadeIn(anchor_label))
        self.wait(2)
        self.play(FadeOut(narration_anchor))

        # ── Shot 5: Show the formula ──
        formula = Text(
            "R* = argmin‖AR − B‖",
            font_size=32, color=ACCENT_PURPLE,
        ).to_edge(DOWN, buff=0.7)

        formula_label = Text(
            "Orthogonal Procrustes: find the rotation matrix R...",
            font_size=18, color=SOFT_WHITE,
        ).next_to(formula, UP, buff=0.2)

        self.play(Write(formula, run_time=1.5))
        self.play(FadeIn(formula_label))
        self.wait(2.5)

        # ── Shot 6: Right grid aligns with left ──
        self.play(FadeOut(formula), FadeOut(formula_label))

        narration_align = Text(
            "...that best aligns the anchor words across spaces.",
            font_size=20, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(narration_align))

        # Animate the right-space words to match the left-space positions
        # (anchors should snap together, others should show their TRUE drift)
        aligned_right_positions = {
            "water": words_left["water"],  # Anchor → same position
            "sun":   words_left["sun"],    # Anchor → same position
            "stone": words_left["stone"],  # Anchor → same position
            "mouse": (2.5, -1.5, GERMANIC_BLUE),  # Shifted → showing drift
            "board": (-0.5, -1.5, GERMANIC_BLUE),  # Shifted → showing some drift
        }

        align_animations = []
        for word, (x, y, _) in aligned_right_positions.items():
            new_pos = right_axes.c2p(x, y)
            align_animations.append(right_dots[word][0].animate.move_to(new_pos))
            label_pos = new_pos + np.array([0.15, 0.15, 0])
            align_animations.append(right_dots[word][1].animate.move_to(label_pos))

        self.play(*align_animations, run_time=3, rate_func=smooth)

        # Show green checkmarks on anchor pairs
        check_marks = VGroup()
        for word in anchor_words:
            mid = (left_dots[word][0].get_center() + right_dots[word][0].get_center()) / 2
            check = Text("✓", font_size=24, color=GREEK_GREEN).move_to(mid)
            check_marks.add(check)

        self.play(FadeIn(check_marks))
        self.wait(1.5)
        self.play(FadeOut(narration_align))

        # ── Shot 7: Show the drift ──
        narration_drift = Text(
            "Now the remaining words reveal their true drift.",
            font_size=20, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(narration_drift))

        # Highlight mouse (displaced) and stone (overlapping)
        mouse_glow = Circle(
            radius=0.3, color=ACCENT_PURPLE,
        ).move_to(right_dots["mouse"][0].get_center())
        mouse_glow.set_fill(ACCENT_PURPLE, opacity=0.1)
        mouse_glow.set_stroke(ACCENT_PURPLE, width=2)

        mouse_drift_label = Text(
            '"mouse" — visibly displaced',
            font_size=16, color=ACCENT_PURPLE,
        ).next_to(mouse_glow, DOWN, buff=0.2)

        stone_check = Text(
            '"stone" — nearly overlapping ✓',
            font_size=16, color=GREEK_GREEN,
        ).next_to(left_dots["stone"][0], DOWN, buff=0.4)

        self.play(
            Create(mouse_glow), FadeIn(mouse_drift_label),
            FadeIn(stone_check),
        )
        self.wait(2.5)

        # ── Shot 8: Final message ──
        self.play(FadeOut(narration_drift))

        final = Text(
            "This is what makes cross-period\ncomparison possible.",
            font_size=32, color=ACCENT_PURPLE,
        ).to_edge(DOWN, buff=0.5)

        self.play(FadeIn(final, shift=UP * 0.2))
        self.wait(3)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1.5,
        )
