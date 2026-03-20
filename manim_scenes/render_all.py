"""
Batch Render Script (Phase 9C)
================================
Renders all 5 Manim scenes to MP4 files.

Usage:
    python manim_scenes/render_all.py                # Render all scenes (720p)
    python manim_scenes/render_all.py --quality high  # 1080p
    python manim_scenes/render_all.py --scene 1       # Render single scene
    python manim_scenes/render_all.py --quality low    # Fast preview (480p)
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Auto-detect ffmpeg on Windows (common installation paths)
FFMPEG_SEARCH_PATHS = [
    r"C:\ffmpeg\ffmpeg-8.0-essentials_build\bin",
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\tools\ffmpeg\bin",
]

for ffmpeg_dir in FFMPEG_SEARCH_PATHS:
    if Path(ffmpeg_dir).exists() and (Path(ffmpeg_dir) / "ffmpeg.exe").exists():
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"  ℹ Added ffmpeg to PATH: {ffmpeg_dir}")
        break

ROOT = Path(__file__).resolve().parent.parent
SCENES_DIR = ROOT / "manim_scenes"
OUTPUT_DIR = ROOT / "results" / "videos"

SCENES = [
    {
        "id": 1,
        "file": "embeddings_intro_scene.py",
        "class": "EmbeddingsIntroScene",
        "title": "What Are Word Embeddings?",
    },
    {
        "id": 2,
        "file": "drift_scene.py",
        "class": "DriftScene",
        "title": "Semantic Drift in Action",
    },
    {
        "id": 3,
        "file": "comparison_scene.py",
        "class": "ComparisonScene",
        "title": "Two Words, Two Stories",
    },
    {
        "id": 4,
        "file": "origin_bars_scene.py",
        "class": "OriginBarsScene",
        "title": "Which Origins Drift Most?",
    },
    {
        "id": 5,
        "file": "alignment_scene.py",
        "class": "AlignmentScene",
        "title": "The Procrustes Alignment",
    },
]

QUALITY_FLAGS = {
    "low": "-ql",       # 480p, 15fps
    "medium": "-qm",    # 720p, 30fps
    "high": "-qh",      # 1080p, 60fps
    "4k": "-qk",        # 4K, 60fps
}


def render_scene(scene: dict, quality: str = "medium"):
    """Render a single Manim scene."""
    file_path = SCENES_DIR / scene["file"]
    q_flag = QUALITY_FLAGS.get(quality, "-qm")

    print(f"\n  ── Rendering Scene {scene['id']}: {scene['title']} ──")
    print(f"    File:    {scene['file']}")
    print(f"    Class:   {scene['class']}")
    print(f"    Quality: {quality}")

    cmd = [
        sys.executable, "-m", "manim", "render",
        q_flag,
        str(file_path),
        scene["class"],
        "--media_dir", str(OUTPUT_DIR),
    ]

    print(f"    Command: {' '.join(cmd[-4:])}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=600,
        )

        if result.returncode == 0:
            print(f"    ✓ Rendered successfully!")
            # Find the output file
            video_files = list(OUTPUT_DIR.rglob("*.mp4"))
            if video_files:
                latest = max(video_files, key=lambda f: f.stat().st_mtime)
                print(f"    ✓ Output: {latest.name}")
        else:
            print(f"    ✗ Rendering failed!")
            if result.stderr:
                # Show last 500 chars of error
                err = result.stderr[-500:]
                print(f"    Error: {err}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"    ✗ Rendering timed out (10 min limit)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Render all Manim scenes")
    parser.add_argument("--quality", choices=["low", "medium", "high", "4k"],
                        default="medium")
    parser.add_argument("--scene", type=int, default=None,
                        help="Render a specific scene (1-5)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Phase 9C: Batch Render Manim Scenes")
    print(f"  Quality: {args.quality}")
    print(f"  Output:  {OUTPUT_DIR}")
    print("=" * 60)

    scenes_to_render = SCENES
    if args.scene:
        scenes_to_render = [s for s in SCENES if s["id"] == args.scene]
        if not scenes_to_render:
            print(f"  ✗ Scene {args.scene} not found")
            return

    success = 0
    for scene in scenes_to_render:
        if render_scene(scene, args.quality):
            success += 1

    print(f"\n{'=' * 60}")
    print(f"  Rendered {success}/{len(scenes_to_render)} scenes")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
