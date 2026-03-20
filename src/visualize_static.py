"""
Static Visualizations (Phase 9A)
==================================
Publication-quality figures using matplotlib and seaborn.

Generates:
  results/figures/drift_bar_chart.png     — Bar chart: origin vs mean drift
  results/figures/drift_box_plot.png      — Box plot: drift distribution per origin
  results/figures/umap_scatter.png        — UMAP scatter: word drift landscape
  results/figures/drift_heatmap.png       — Heatmap: per-word drift by origin

Usage:
    python src/visualize_static.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS_DIR, ORIGIN_CLASSES

FIGURES_DIR = RESULTS_DIR / "figures"
DRIFT_CSV = RESULTS_DIR / "drift_scores.csv"
UMAP_CSV = RESULTS_DIR / "umap_coords.csv"
ORIGIN_CSV = RESULTS_DIR / "origin_drift_summary.csv"

# ── Design palette (consistent with Manim scenes) ──
COLORS = {
    "Germanic": "#4FC3F7",
    "Latin":    "#EF5350",
    "Greek":    "#66BB6A",
    "Other":    "#FFA726",
}
BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4e"


def setup_style():
    """Set up publication-quality dark theme."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
    })


def plot_drift_bar_chart(df: pd.DataFrame, origin_summary: pd.DataFrame):
    """Bar chart: mean drift by origin with error bars and individual points."""
    fig, ax = plt.subplots(figsize=(10, 6))

    origins = origin_summary.sort_values("mean_drift")
    x = range(len(origins))

    # Bars
    bars = ax.bar(
        x,
        origins["mean_drift"],
        yerr=origins["std_drift"],
        capsize=5,
        color=[COLORS[o] for o in origins["origin_class"]],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        error_kw={"elinewidth": 1.5, "ecolor": TEXT_COLOR, "alpha": 0.7},
    )

    # Overlay individual word dots (beeswarm-like)
    np.random.seed(42)
    for i, (_, row) in enumerate(origins.iterrows()):
        origin = row["origin_class"]
        word_drifts = df[df["origin_class"] == origin]["drift_score"]
        jitter = np.random.uniform(-0.15, 0.15, len(word_drifts))
        ax.scatter(
            [i + j for j in jitter],
            word_drifts,
            color=COLORS[origin],
            edgecolor="white",
            s=30,
            alpha=0.7,
            zorder=5,
            linewidth=0.5,
        )

    # Value labels on bars
    for i, (_, row) in enumerate(origins.iterrows()):
        ax.text(i, row["mean_drift"] + row["std_drift"] + 0.02,
                f"{row['mean_drift']:.3f}",
                ha="center", fontsize=11, fontweight="bold", color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(origins["origin_class"], fontsize=13)
    ax.set_ylabel("Semantic Drift Score", fontsize=13)
    ax.set_title("Mean Semantic Drift by Etymological Origin", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.2)

    # Significance annotation
    ax.annotate("", xy=(0, 0.92), xytext=(3, 0.92),
                arrowprops=dict(arrowstyle="-", lw=1, color=TEXT_COLOR))
    ax.text(1.5, 0.94, "p = 0.061 (Kruskal-Wallis)", ha="center", fontsize=10,
            color=TEXT_COLOR, alpha=0.8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "drift_bar_chart.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ drift_bar_chart.png")


def plot_drift_box_plot(df: pd.DataFrame):
    """Box plot showing drift distribution per origin."""
    fig, ax = plt.subplots(figsize=(10, 6))

    order = df.groupby("origin_class")["drift_score"].mean().sort_values().index.tolist()

    # Custom box plot
    bp = ax.boxplot(
        [df[df["origin_class"] == o]["drift_score"].values for o in order],
        labels=order,
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker="o", markerfacecolor=TEXT_COLOR, markersize=4, alpha=0.5),
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=TEXT_COLOR, linewidth=1),
        capprops=dict(color=TEXT_COLOR, linewidth=1),
    )

    for patch, origin in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS[origin])
        patch.set_alpha(0.7)
        patch.set_edgecolor("white")

    # Overlay strip plot
    np.random.seed(42)
    for i, origin in enumerate(order, 1):
        vals = df[df["origin_class"] == origin]["drift_score"].values
        jitter = np.random.uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            i + jitter, vals,
            color=COLORS[origin], edgecolor="white",
            s=25, alpha=0.6, zorder=5, linewidth=0.5,
        )

    ax.set_ylabel("Semantic Drift Score", fontsize=13)
    ax.set_title("Drift Distribution by Etymological Origin", fontsize=16, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "drift_box_plot.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ drift_box_plot.png")


def plot_umap_scatter(umap_df: pd.DataFrame):
    """UMAP scatter: words in 2D embedding space with drift arrows."""
    fig, ax = plt.subplots(figsize=(14, 10))

    for origin in ORIGIN_CLASSES:
        og = umap_df[umap_df["origin_class"] == origin]
        color = COLORS[origin]

        # Old positions (hollow circles)
        ax.scatter(og["x_old"], og["y_old"], facecolors="none",
                  edgecolors=color, s=60, linewidth=1.5, alpha=0.6,
                  label=f"{origin} (1800s)", zorder=3)

        # New positions (filled circles)
        ax.scatter(og["x_new"], og["y_new"], c=color, s=60, alpha=0.8,
                  edgecolors="white", linewidth=0.5, zorder=4)

        # Arrows from old → new
        for _, row in og.iterrows():
            ax.annotate(
                "", xy=(row["x_new"], row["y_new"]),
                xytext=(row["x_old"], row["y_old"]),
                arrowprops=dict(
                    arrowstyle="->", color=color, alpha=0.4,
                    lw=1, connectionstyle="arc3,rad=0.1",
                ),
            )

    # Label top-8 highest-drift words
    top_drift = umap_df.nlargest(8, "drift_score")
    for _, row in top_drift.iterrows():
        ax.annotate(
            f"{row['word']} ({row['drift_score']:.2f})",
            xy=(row["x_new"], row["y_new"]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=9, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BG_COLOR, alpha=0.8,
                     edgecolor=COLORS[row["origin_class"]]),
        )

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title("Semantic Drift Landscape (UMAP Projection)",
                fontsize=16, fontweight="bold")

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for origin in ORIGIN_CLASSES:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[origin],
                  markersize=10, label=f"{origin} (filled = modern)")
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
              markeredgecolor=TEXT_COLOR, markersize=10, label="Hollow = historical")
    )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10,
             facecolor=BG_COLOR, edgecolor=GRID_COLOR)

    ax.grid(alpha=0.15)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "umap_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ umap_scatter.png")


def plot_drift_heatmap(df: pd.DataFrame):
    """Heatmap showing per-word drift scores grouped by origin."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Sort words within each origin by drift
    df_sorted = df.sort_values(["origin_class", "drift_score"], ascending=[True, False])

    # Create wide-form data for heatmap display
    words = df_sorted["word"].tolist()
    drifts = df_sorted["drift_score"].tolist()
    origins = df_sorted["origin_class"].tolist()

    # Color-coded bar chart sorted by drift within origin
    colors = [COLORS[o] for o in origins]
    bars = ax.barh(range(len(words)), drifts, color=colors, alpha=0.85, edgecolor="none")

    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=7)
    ax.set_xlabel("Semantic Drift Score")
    ax.set_title("Per-Word Drift Scores (Grouped by Origin)", fontsize=16, fontweight="bold")

    # Origin group separators
    prev = None
    for i, origin in enumerate(origins):
        if origin != prev:
            if prev is not None:
                ax.axhline(y=i - 0.5, color=TEXT_COLOR, alpha=0.3, lw=0.5)
            prev = origin

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=COLORS[o], label=o) for o in ORIGIN_CLASSES]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10,
             facecolor=BG_COLOR, edgecolor=GRID_COLOR)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "drift_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ drift_heatmap.png")


def main():
    print("=" * 60)
    print("  Phase 9A: Static Visualizations")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    # Load data
    drift_df = pd.read_csv(DRIFT_CSV)
    valid = drift_df[drift_df["status"] == "OK"].copy()
    origin_summary = pd.read_csv(ORIGIN_CSV)
    umap_df = pd.read_csv(UMAP_CSV)

    print(f"\n  Generating figures to {FIGURES_DIR.name}/...")

    plot_drift_bar_chart(valid, origin_summary)
    plot_drift_box_plot(valid)
    plot_umap_scatter(umap_df)
    plot_drift_heatmap(valid)

    print(f"\n  ✓ All static figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
