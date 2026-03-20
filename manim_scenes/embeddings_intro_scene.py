"""
Scene 1: "What Are Word Embeddings?" (Educational Intro)
=========================================================
3Blue1Brown-style animation introducing the concept of word embeddings
and semantic drift. Builds incrementally, reveals one concept at a time.
"""

import sys
from pathlib import Path
from manim import *

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── 3B1B Color palette ──
DEEP_BG = "#1a1a2e"
GERMANIC_BLUE = "#4FC3F7"
LATIN_RED = "#EF5350"
GREEK_GREEN = "#66BB6A"
OTHER_AMBER = "#FFA726"
SOFT_WHITE = "#e0e0e0"
ACCENT_PURPLE = "#BB86FC"
GRID_DIM = "#2a2a4e"


class EmbeddingsIntroScene(Scene):
    def construct(self):
        self.camera.background_color = DEEP_BG

        # ── Shot 1: Title ──
        title = Text(
            "How do words live in\nmathematical space?",
            font_size=48, color=SOFT_WHITE,
        ).move_to(ORIGIN)
        subtitle = Text(
            "An introduction to word embeddings",
            font_size=24, color=ACCENT_PURPLE,
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(title, run_time=2))
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ── Shot 2: Word → Vector ──
        word_text = Text('"king"', font_size=52, color=GERMANIC_BLUE)
        word_text.move_to(ORIGIN)
        self.play(Write(word_text))
        self.wait(0.5)

        vec_text = Text(
            "v_king = [0.21, -0.53, 0.81, ...]",
            font_size=32, color=SOFT_WHITE,
        ).move_to(ORIGIN)

        narration1 = Text(
            "Every word becomes a vector — a list of numbers.",
            font_size=22, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.7)

        self.play(
            ReplacementTransform(word_text, vec_text),
            FadeIn(narration1, shift=UP * 0.2),
        )
        self.wait(2)
        self.play(FadeOut(vec_text), FadeOut(narration1))

        # ── Shot 3: 2D Axes with word dots ──
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6, y_length=5,
            tips=False,
            axis_config={"color": GRID_DIM, "stroke_width": 1},
        ).shift(UP * 0.2)

        # Word positions (illustrative 2D)
        word_positions = {
            "king":  (1.5, 2.0, GERMANIC_BLUE),
            "queen": (1.8, 1.2, LATIN_RED),
            "man":   (-0.5, 1.8, GERMANIC_BLUE),
            "woman": (-0.2, 1.0, LATIN_RED),
            "dog":   (-1.5, -1.5, GREEK_GREEN),
            "cat":   (-1.0, -1.2, GREEK_GREEN),
        }

        dots = {}
        labels = {}
        for word, (x, y, color) in word_positions.items():
            pos = axes.c2p(x, y)
            dot = Dot(pos, radius=0.1, color=color)
            dot.set_glow_factor(0.3)
            label = Text(word, font_size=18, color=color).next_to(dot, UR, buff=0.1)
            dots[word] = dot
            labels[word] = label

        narration2 = Text(
            "Words with similar meanings cluster together.",
            font_size=22, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.7)

        self.play(Create(axes, run_time=1))
        # Staggered FadeIn for each word
        for word in ["king", "queen", "man", "woman", "dog", "cat"]:
            self.play(
                FadeIn(dots[word], scale=0.5),
                FadeIn(labels[word]),
                run_time=0.4,
            )
        self.play(FadeIn(narration2, shift=UP * 0.2))
        self.wait(2)

        # ── Shot 4: The famous analogy ──
        self.play(FadeOut(narration2))

        # king - man + woman ≈ queen
        arrow_km = Arrow(
            dots["king"].get_center(), dots["man"].get_center(),
            color=ACCENT_PURPLE, stroke_width=2, buff=0.15,
        )
        arrow_wq = Arrow(
            dots["woman"].get_center(), dots["queen"].get_center(),
            color=ACCENT_PURPLE, stroke_width=2, buff=0.15,
        )
        # Dashed parallel lines
        dash_kw = DashedLine(
            dots["king"].get_center(), dots["woman"].get_center(),
            color=SOFT_WHITE, stroke_width=1,
        ).set_opacity(0.4)
        dash_mq = DashedLine(
            dots["man"].get_center(), dots["queen"].get_center(),
            color=SOFT_WHITE, stroke_width=1,
        ).set_opacity(0.4)

        formula = Text(
            "king − man + woman ≈ queen",
            font_size=26, color=ACCENT_PURPLE,
        ).to_edge(DOWN, buff=0.7)

        self.play(
            Create(arrow_km), Create(arrow_wq),
            Create(dash_kw), Create(dash_mq),
            run_time=1.5,
        )
        self.play(Write(formula))
        self.wait(2.5)

        # ── Shot 5: Dimensionality note ──
        self.play(FadeOut(formula))
        dim_note = Text(
            "This is a 100-dimensional space, projected to 2D for you.",
            font_size=20, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.7).set_opacity(0.7)
        self.play(FadeIn(dim_note))
        self.wait(1.5)

        # ── Shot 6: Transition to time periods ──
        self.play(
            FadeOut(dim_note),
            FadeOut(arrow_km), FadeOut(arrow_wq),
            FadeOut(dash_kw), FadeOut(dash_mq),
            *[FadeOut(labels[w]) for w in ["dog", "cat", "king", "queen", "man", "woman"]],
            *[FadeOut(dots[w]) for w in ["dog", "cat", "king", "queen", "man", "woman"]],
        )

        transition_text = Text(
            "But what if we train on text\nfrom different centuries?",
            font_size=36, color=SOFT_WHITE,
        ).move_to(ORIGIN)
        self.play(Write(transition_text, run_time=1.5))
        self.wait(1.5)
        self.play(FadeOut(transition_text))

        # ── Shot 7: Mouse drift demo ──
        # 1800s: mouse near "rat", "trap", "vermin"
        pos_mouse_old = axes.c2p(-1.5, 1.5)
        pos_rat = axes.c2p(-1.8, 1.8)
        pos_trap = axes.c2p(-1.2, 2.0)
        pos_vermin = axes.c2p(-2.0, 1.2)

        # 2000s: mouse near "click", "computer", "cursor"
        pos_mouse_new = axes.c2p(1.5, -1.0)
        pos_click = axes.c2p(1.8, -0.7)
        pos_computer = axes.c2p(1.2, -1.5)
        pos_cursor = axes.c2p(2.0, -1.2)

        era_label_old = Text("1800s", font_size=28, color=OTHER_AMBER).to_edge(UP, buff=0.5)

        mouse_dot = Dot(pos_mouse_old, radius=0.12, color=GERMANIC_BLUE)
        mouse_dot.set_glow_factor(0.5)
        mouse_label = Text("mouse", font_size=22, color=GERMANIC_BLUE).next_to(mouse_dot, DOWN, buff=0.15)

        old_neighbors = VGroup(
            *[Dot(p, radius=0.07, color=GRID_DIM).set_opacity(0.6) for p in [pos_rat, pos_trap, pos_vermin]],
        )
        old_nbr_labels = VGroup(
            Text("rat", font_size=14, color=SOFT_WHITE).move_to(pos_rat + UP * 0.2),
            Text("trap", font_size=14, color=SOFT_WHITE).move_to(pos_trap + UP * 0.2),
            Text("vermin", font_size=14, color=SOFT_WHITE).move_to(pos_vermin + LEFT * 0.4),
        ).set_opacity(0.7)

        # Connecting lines
        old_lines = VGroup(
            *[Line(pos_mouse_old, p, color=GERMANIC_BLUE, stroke_width=1).set_opacity(0.3)
              for p in [pos_rat, pos_trap, pos_vermin]]
        )

        self.play(
            FadeIn(era_label_old),
            FadeIn(mouse_dot, scale=0.5),
            FadeIn(mouse_label),
        )
        self.play(
            FadeIn(old_neighbors), FadeIn(old_nbr_labels),
            Create(old_lines),
        )

        narration_old = Text(
            "In the 1800s, 'mouse' lived among rodents.",
            font_size=20, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(narration_old))
        self.wait(2)

        # Transition to 2000s
        era_label_new = Text("2000s", font_size=28, color=LATIN_RED).to_edge(UP, buff=0.5)

        new_neighbors = VGroup(
            *[Dot(p, radius=0.07, color=GRID_DIM).set_opacity(0.6) for p in [pos_click, pos_computer, pos_cursor]],
        )
        new_nbr_labels = VGroup(
            Text("click", font_size=14, color=SOFT_WHITE).move_to(pos_click + UP * 0.2),
            Text("computer", font_size=14, color=SOFT_WHITE).move_to(pos_computer + DOWN * 0.25),
            Text("cursor", font_size=14, color=SOFT_WHITE).move_to(pos_cursor + RIGHT * 0.4),
        ).set_opacity(0.7)

        new_lines = VGroup(
            *[Line(pos_mouse_new, p, color=LATIN_RED, stroke_width=1).set_opacity(0.3)
              for p in [pos_click, pos_computer, pos_cursor]]
        )

        # Trail line from old to new position
        trail = DashedLine(
            pos_mouse_old, pos_mouse_new,
            color=ACCENT_PURPLE, stroke_width=2,
        ).set_opacity(0.4)

        narration_new = Text(
            "In the 2000s, 'mouse' moved to the world of computing.",
            font_size=20, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.7)

        self.play(
            ReplacementTransform(era_label_old, era_label_new),
            FadeOut(narration_old),
            # Fade out old context
            FadeOut(old_neighbors), FadeOut(old_nbr_labels), FadeOut(old_lines),
        )
        self.play(
            mouse_dot.animate.move_to(pos_mouse_new),
            mouse_label.animate.next_to(Dot(pos_mouse_new), DOWN, buff=0.15),
            Create(trail),
            run_time=2,
        )
        self.play(
            FadeIn(new_neighbors), FadeIn(new_nbr_labels),
            Create(new_lines),
            FadeIn(narration_new),
        )
        self.wait(2.5)

        # ── Shot 8: "This is Semantic Drift" ──
        self.play(
            FadeOut(narration_new),
            FadeOut(new_neighbors), FadeOut(new_nbr_labels),
            FadeOut(new_lines), FadeOut(mouse_dot), FadeOut(mouse_label),
            FadeOut(trail), FadeOut(axes), FadeOut(era_label_new),
        )

        final_title = Text(
            "This is Semantic Drift.",
            font_size=52, color=ACCENT_PURPLE,
        ).move_to(ORIGIN)

        final_subtitle = Text(
            "The same word. A different era. A different meaning.",
            font_size=22, color=SOFT_WHITE,
        ).next_to(final_title, DOWN, buff=0.5)

        self.play(Write(final_title, run_time=1.5))
        self.play(FadeIn(final_subtitle, shift=UP * 0.2))
        self.wait(3)
        self.play(FadeOut(final_title), FadeOut(final_subtitle))
