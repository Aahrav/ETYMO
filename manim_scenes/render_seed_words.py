"""
Batch Render Script for Benchmark Seed Words (ManimGX)
======================================================
Pre-renders all curated seed words with documented historical shifts
using GPU-accelerated ManimGX, storing them in results/videos/videos/words/
for instant playback in the Web Explorer.
"""

import sys
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENES_DIR = ROOT / "manim_scenes"
OUTPUT_DIR = ROOT / "results" / "videos" / "videos" / "words"

# Benchmark seed words + common exploratory targets
SEED_WORDS_LIST = [
    # Germanic
    "craft", "mouse", "web", "stream", "cloud", "ship", "board", "barn",
    # Latin
    "virus", "cell", "text", "monitor", "cabinet", "culture", "tablet", "journal",
    # Greek
    "atom", "energy", "organ",
    # Sanskrit & Hindi
    "avatar", "nirvana", "yoga", "punch", "sugar",
    # PIE & Invariant Anchors
    "father", "mother", "water", "heart", "night", "star", "eye", "name", "new", "three",
    # Other & Cross-lingual
    "algorithm", "candy", "coffee", "magazine", "tsunami", "freedom",
]


def render_word_video(word: str) -> bool:
    word = word.lower().strip()
    scene_file = SCENES_DIR / "single_word_scene.py"
    scratch_file = ROOT / f"_tmp_batch_{word}.py"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"{word}.mp4"

    content = scene_file.read_text(encoding="utf-8")
    content = content.replace('target_word = "craft"', f'target_word = "{word}"')
    scratch_file.write_text(content, encoding="utf-8")

    manimgx_bin = shutil.which("manimgx") or "/home/aahrav/.local/bin/manimgx"
    cmd = [
        str(manimgx_bin), "render",
        "-r", "1280x720",
        "--fps", "30",
        "-o", str(out_file),
        str(scratch_file),
        "SingleWordDriftScene",
    ]

    try:
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, timeout=60, text=True)
        elapsed = time.time() - t0

        if result.returncode == 0 and out_file.exists():
            size_kb = out_file.stat().st_size / 1024
            print(f"  ✓ {word:<12} (720p, {size_kb:.0f} KB) in {elapsed:.2f}s")
            return True
        else:
            print(f"  ✗ {word:<12} failed (code {result.returncode})")
            return False
    except Exception as e:
        print(f"  ✗ {word:<12} error: {e}")
        return False
    finally:
        if scratch_file.exists():
            scratch_file.unlink()


def main():
    print("=" * 60)
    print("  Pre-rendering Benchmark Seed Word Animations (ManimGX)")
    print(f"  Targets: {len(SEED_WORDS_LIST)} words across 6 etymological origins")
    print("=" * 60)

    t_start = time.time()
    succeeded = 0

    for i, word in enumerate(SEED_WORDS_LIST, 1):
        print(f"[{i:02d}/{len(SEED_WORDS_LIST):02d}] Rendering '{word}'...", end="", flush=True)
        print("\r", end="")
        if render_word_video(word):
            succeeded += 1

    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  Batch complete: {succeeded}/{len(SEED_WORDS_LIST)} videos ready in {total_time:.1f}s")
    print(f"  Average speed: {total_time / max(succeeded, 1):.2f}s per video")
    print("=" * 60)


if __name__ == "__main__":
    main()
