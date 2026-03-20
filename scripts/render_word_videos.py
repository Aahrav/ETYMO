"""
Batch Render Word Videos (Phase 13)
======================================
Renders Manim animations for the 'Top' words of each origin.
Uses ComparisonScene with parametric word selection.
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
VIDEO_DIR = RESULTS_DIR / "videos" / "videos" / "words"
SELECTED_WORDS_CSV = ROOT / "data" / "selected_words.csv"
SCENE_FILE = ROOT / "manim_scenes" / "comparison_scene.py"

# Custom CSS-like colors for logging
GREEN = "\033[92m"
BLUE = "\033[94m"
ENDC = "\033[0m"

def render_word_video(word, output_name=None):
    """Render a 480p 15fps (low quality, fast) video for a single word."""
    if not output_name:
        output_name = word
        
    dest = VIDEO_DIR / f"{output_name}.mp4"
    if dest.exists():
        print(f"{GREEN}  [Skipping] {word} (Already exists){ENDC}")
        return
        
    print(f"{BLUE}  [Rendering] {word}...{ENDC}")
    
    scratch_file = ROOT / "tmp_scene.py"
    content = SCENE_FILE.read_text(encoding="utf-8")
    
    # Replace default words
    content = content.replace('word_a = "craft"', f'word_a = "{word}"')
    # Use 'mother' as a stable reference for comparison if word itself is drifted
    content = content.replace('word_b = "room"', f'word_b = "mother"')
    
    scratch_file.write_text(content, encoding="utf-8")
    
    cmd = [
        sys.executable, "-m", "manim", str(scratch_file), "ComparisonScene",
        "-ql",  # Low quality for speed
        "--media_dir", str(RESULTS_DIR / "videos"),
        "-o", output_name
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # Manim outputs to: results/videos/videos/tmp_scene/480p15/output_name.mp4
        # We want to move it to: results/videos/words/output_name.mp4
        source = RESULTS_DIR / "videos" / "videos" / "tmp_scene" / "480p15" / f"{output_name}.mp4"
        dest = VIDEO_DIR / f"{output_name}.mp4"
        
        if source.exists():
            import shutil
            shutil.copy2(source, dest)
            print(f"{GREEN}    ✓ Success: {output_name}.mp4{ENDC}")
        else:
            print(f"    ✗ Error: Could not find output at {source}")
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Error rendering {word}: {e.stderr.decode()}")
    finally:
        if scratch_file.exists():
            scratch_file.unlink()

def main():
    if not SELECTED_WORDS_CSV.exists():
        print(f"✗ {SELECTED_WORDS_CSV} not found. Run word_selector.py first.")
        return

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(SELECTED_WORDS_CSV)
    
    # Pick Top 3 per origin by confidence/drift
    origins = df["origin_class"].unique()
    to_render = []
    
    for org in origins:
        # Get 1 top word for this origin
        subset = df[df["origin_class"] == org].head(1)
        to_render.extend(subset["word"].tolist())
    
    print(f"Starting batch render for {len(to_render)} words...")
    
    for word in to_render:
        render_word_video(word)
        
    print("\n✓ Batch rendering complete!")
    print(f"Videos saved to: {VIDEO_DIR}")

if __name__ == "__main__":
    main()
