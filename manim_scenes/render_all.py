"""
Batch Render Script for Diachronic Semantic Shift Animations (ManimGX)
======================================================================
Renders all 5 Manim scenes to MP4 files using GPU-accelerated ManimGX.

Usage:
    python manim_scenes/render_all.py                # Render all 5 scenes (720p, 30fps)
    python manim_scenes/render_all.py --scene 1       # Render specific scene (1-5)
    python manim_scenes/render_all.py --quality high  # 1080p, 30fps
"""

import sys
import os
import shutil
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENES_DIR = ROOT / "manim_scenes"
OUTPUT_DIR = ROOT / "results" / "videos"

SCENES = [
    {
        "id": 1,
        "file": "embeddings_intro_scene.py",
        "class": "EmbeddingsIntroScene",
        "title": "What Are Word Embeddings?",
        "out_dir": OUTPUT_DIR / "embeddings_intro_scene" / "720p30",
        "filename": "EmbeddingsIntroScene.mp4",
    },
    {
        "id": 2,
        "file": "drift_scene.py",
        "class": "DriftScene",
        "title": "Semantic Drift in Action",
        "out_dir": OUTPUT_DIR / "drift_scene" / "720p30",
        "filename": "DriftScene.mp4",
    },
    {
        "id": 3,
        "file": "comparison_scene.py",
        "class": "ComparisonScene",
        "title": "Two Words, Two Stories",
        "out_dir": OUTPUT_DIR / "comparison_scene" / "720p30",
        "filename": "ComparisonScene.mp4",
    },
    {
        "id": 4,
        "file": "origin_bars_scene.py",
        "class": "OriginBarsScene",
        "title": "Which Origins Drift Most?",
        "out_dir": OUTPUT_DIR / "origin_bars_scene" / "720p30",
        "filename": "OriginBarsScene.mp4",
    },
    {
        "id": 5,
        "file": "alignment_scene.py",
        "class": "AlignmentScene",
        "title": "The Procrustes Alignment",
        "out_dir": OUTPUT_DIR / "alignment_scene" / "720p30",
        "filename": "AlignmentScene.mp4",
    },
]

RESOLUTIONS = {
    "low": "854x480",
    "medium": "1280x720",
    "high": "1920x1080",
    "4k": "3840x2160",
}


def render_scene(scene: dict, quality: str = "medium") -> bool:
    """Render a single ManimGX scene."""
    file_path = SCENES_DIR / scene["file"]
    scene["out_dir"].mkdir(parents=True, exist_ok=True)
    out_file = scene["out_dir"] / scene["filename"]
    res = RESOLUTIONS.get(quality, "1280x720")

    print(f"\n  ── Rendering Scene {scene['id']}: {scene['title']} ──")
    print(f"    File:       {scene['file']}")
    print(f"    Class:      {scene['class']}")
    print(f"    Resolution: {res} @ 30fps")
    print(f"    Target:     {out_file.relative_to(ROOT)}")

    manimgx_bin = shutil.which("manimgx") or "/home/aahrav/.local/bin/manimgx"
    if Path(manimgx_bin).exists():
        cmd = [
            str(manimgx_bin), "render",
            "-r", res,
            "--fps", "30",
            "-o", str(out_file),
            str(file_path),
            scene["class"],
        ]
    else:
        print("    [Warning] manimgx not found, falling back to legacy manim")
        cmd = [
            sys.executable, "-m", "manim", "render",
            "-qm",
            str(file_path),
            scene["class"],
            "--media_dir", str(OUTPUT_DIR),
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=120,
        )

        if result.returncode == 0 and out_file.exists():
            size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"    ✓ Successfully rendered in ManimGX ({size_mb:.2f} MB)")
            return True
        else:
            print(f"    ✗ Rendering failed (code {result.returncode})")
            if result.stderr:
                print(f"    Error: {result.stderr[-400:]}")
            if result.stdout:
                print(f"    Output: {result.stdout[-400:]}")
            return False

    except subprocess.TimeoutExpired:
        print("    ✗ Rendering timed out")
        return False


def main():
    parser = argparse.ArgumentParser(description="Render all Manim scenes using ManimGX")
    parser.add_argument("--quality", choices=["low", "medium", "high", "4k"], default="medium")
    parser.add_argument("--scene", type=int, default=None, help="Render a specific scene (1-5)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ETYMO ManimGX High-Performance Scene Renderer")
    print(f"  Quality: {args.quality} ({RESOLUTIONS.get(args.quality)})")
    print("=" * 60)

    if args.scene is not None:
        targets = [s for s in SCENES if s["id"] == args.scene]
        if not targets:
            print(f"  Error: Scene {args.scene} not found (must be 1-5)")
            sys.exit(1)
    else:
        targets = SCENES

    success_count = 0
    for scene in targets:
        if render_scene(scene, quality=args.quality):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"  Render complete: {success_count}/{len(targets)} scenes succeeded.")
    print("=" * 60)


if __name__ == "__main__":
    main()
