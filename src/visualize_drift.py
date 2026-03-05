from manim import *

import pandas as pd
import numpy as np
import os

# Placeholder for data
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'drift_data.csv')

class SemanticDriftScene(Scene):
    def construct(self):
        # 1. Setup Title
        title = Text("Semantic Drift Analysis", font_size=40).to_edge(UP)
        self.play(Write(title))
        
        # 2. Add Legend
        legend_germanic = Dot(color=BLUE).next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=1)
        text_germanic = Text("Germanic", font_size=24).next_to(legend_germanic, RIGHT)
        
        legend_latin = Dot(color=RED).next_to(legend_germanic, DOWN, buff=0.2)
        text_latin = Text("Latin", font_size=24).next_to(legend_latin, RIGHT)
        
        self.play(FadeIn(legend_germanic), Write(text_germanic), 
                  FadeIn(legend_latin), Write(text_latin))
        
        # 3. Load Data (Mock if not present)
        # Assuming data has columns: word, x1, y1, x2, y2, origin
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception:
            # Mock data
            data = {
                'word': ['hound', 'dog', 'starve', 'meat'],
                'x1': [-2, -1, 2, 1], 'y1': [1, -2, 1, -1],
                'x2': [-2.2, -0.8, 2.5, 0.5], 'y2': [1.1, -1.8, 0.5, -0.5],
                'origin': ['Germanic', 'Germanic', 'Germanic', 'Germanic']
            }
            df = pd.DataFrame(data)
            
        dots = VGroup()
        labels = VGroup()
        
        # 4. Create Initial State (T1)
        for _, row in df.iterrows():
            color = BLUE if row['origin'] == 'Germanic' else RED
            dot = Dot(point=[row['x1'], row['y1'], 0], color=color)
            label = Text(row['word'], font_size=20).next_to(dot, UP, buff=0.1)
            
            dots.add(dot)
            labels.add(label)
            
        self.play(Create(dots), Write(labels))
        self.wait(1)
        
        # 5. Animate Drift (T1 -> T2)
        animations = []
        for i, row in df.iterrows():
            dot = dots[i]
            label = labels[i]
            
            target_point = [row['x2'], row['y2'], 0]
            
            # Animate dot movement
            animations.append(dot.animate.move_to(target_point))
            
            # Animate label movement
            animations.append(label.animate.next_to(target_point, UP, buff=0.1))
            
            # Trace path
            path = Line(start=[row['x1'], row['y1'], 0], end=target_point, color=YELLOW, stroke_opacity=0.5)
            self.add(path) # Add path immediately or animate creation?
            # Let's just add it as a trace
            
        self.play(*animations, run_time=3)
        self.wait(2)
