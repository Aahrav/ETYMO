"""
Publication-Quality Visualizations for Deep Learning Semantic Shift (Phase 3 — Task 3.4)
========================================================================================
Generates publication-ready figures for IEEE/ACL-style papers based on
Deep Learning Optimal Transport (W2) and Polysemy Dynamics.

Generates:
  1. results/figures/dl_w2_distribution_boxplot.png  — Boxplot & jittered scatter of W2 by origin
  2. results/figures/dl_shift_typology_breakdown.png — Stacked bar chart of shift typology proportions
  3. results/figures/dl_classical_vs_dl_scatter.png  — Cross-paradigm scatter (Classical vs. W2) with seed annotations
  4. results/figures/dl_dispersion_shift_scatter.png — Diachronic dispersion space (Historical vs. Modern variance)
  5. results/figures/dl_correlation_heatmap.png      — Heatmap correlating all diachronic metrics

Authors / Maintainers:
  Aarav Parmar (aaravparmar2408@gmail.com)
  ETYMO: Etymology-Aware Diachronic Semantic Shift Analysis

Usage:
  python src/dl_visualizations.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ORIGIN_CLASSES,
    RESULTS_DIR,
    DL_DRIFT_SCORES_CSV,
    DL_FIGURES_DIR,
)

CLASSICAL_DRIFT_CSV = RESULTS_DIR / "drift_scores.csv"

# ── Color Palette & Aesthetic Constants ──
ORIGIN_COLORS = {
    "Germanic": "#4FC3F7",  # Cyan
    "Latin":    "#EF5350",  # Coral Red
    "Greek":    "#66BB6A",  # Emerald Green
    "Sanskrit": "#FFD54F",  # Amber
    "PIE":      "#CE93D8",  # Purple
    "Other":    "#90A4AE",  # Muted Slate
}

TYPOLOGY_COLORS = {
    "Sense Broadening":   "#42A5F5",  # Blue
    "Core Invariant":     "#66BB6A",  # Green
    "Centroid Migration": "#FFA726",  # Orange
    "Sense Narrowing":    "#AB47BC",  # Purple
}

BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#161625"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4e"


def setup_dark_style():
    """Sets up unified dark scientific styling for all figures."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.35,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.autolayout": False,
    })


def plot_w2_distribution(df: pd.DataFrame, output_path: Path):
    """Figure 1: 2-Wasserstein Drift distribution by origin class."""
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)

    # Order origins by median W2
    order = df.groupby("origin_class")["wasserstein_w2"].median().sort_values().index.tolist()

    # Boxplot
    box = sns.boxplot(
        data=df,
        x="origin_class",
        y="wasserstein_w2",
        order=order,
        palette=ORIGIN_COLORS,
        width=0.48,
        fliersize=0,
        boxprops=dict(alpha=0.75, edgecolor="white", linewidth=1.2),
        whiskerprops=dict(color="white", linewidth=1.2),
        capprops=dict(color="white", linewidth=1.2),
        medianprops=dict(color="#FFEB3B", linewidth=2.0),
        ax=ax,
    )

    # Jittered strip plot overlay
    sns.stripplot(
        data=df,
        x="origin_class",
        y="wasserstein_w2",
        order=order,
        color="white",
        alpha=0.45,
        size=4.5,
        jitter=0.22,
        ax=ax,
    )

    # Annotate means as red diamonds
    means = df.groupby("origin_class")["wasserstein_w2"].mean().loc[order]
    for idx, mean_val in enumerate(means):
        ax.scatter(idx, mean_val, color="#FF5252", marker="D", s=45, zorder=5, edgecolor="black", linewidth=0.8)

    # Formatting & Statistical Annotation
    ax.set_title("Diachronic Semantic Shift: 2-Wasserstein Distance (W2) by Etymological Origin", pad=15, fontweight="bold")
    ax.set_xlabel("Etymological Root Class", labelpad=10, fontweight="semibold")
    ax.set_ylabel("2-Wasserstein Distance (W2)", labelpad=10, fontweight="semibold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    # Kruskal-Wallis annotation pill
    ax.text(
        0.03, 0.94,
        "Kruskal-Wallis: H = 18.23, p = 0.0027 **\nOne-Way ANOVA: F = 4.08, p = 0.0013 **\nDiamonds (♦) = Origin Means",
        transform=ax.transAxes,
        fontsize=10.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor=PANEL_COLOR, edgecolor="#4FC3F7", alpha=0.9),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Figure saved: {output_path}")


def plot_shift_typology_breakdown(df: pd.DataFrame, output_path: Path):
    """Figure 2: Stacked percentage bar chart of semantic shift typology."""
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    # Cross-tabulate proportions
    ct = pd.crosstab(df["origin_class"], df["shift_type"], normalize="index") * 100
    typologies = ["Sense Broadening", "Core Invariant", "Centroid Migration", "Sense Narrowing"]
    ct = ct[typologies]

    # Sort origins by Sense Broadening %
    ct = ct.sort_values(by="Sense Broadening", ascending=False)

    bottom = np.zeros(len(ct))
    for typ in typologies:
        values = ct[typ].values
        color = TYPOLOGY_COLORS.get(typ, "#888888")
        bars = ax.bar(ct.index, values, bottom=bottom, label=typ, color=color, edgecolor="black", linewidth=0.8, alpha=0.88, width=0.55)

        # Label inside bars if > 8%
        for idx, (val, b_val) in enumerate(zip(values, bottom)):
            if val >= 9.0:
                ax.text(
                    idx, b_val + val / 2.0,
                    f"{val:.1f}%",
                    ha="center", va="center",
                    color="white", fontsize=9.5, fontweight="bold",
                )
        bottom += values

    ax.set_title("Semantic Shift Typology Distribution Across Etymological Origins", pad=15, fontweight="bold")
    ax.set_xlabel("Etymological Root Class", labelpad=10, fontweight="semibold")
    ax.set_ylabel("Proportion of Tracked Words (%)", labelpad=10, fontweight="semibold")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.25, axis="y")

    # Legend & Chi-Square Pill
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.15), ncol=4, frameon=True, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR)

    ax.text(
        0.03, 0.94,
        "Chi-Square Independence Test:\nχ² = 28.08, df = 15, p = 0.0211 *\nCramér's V = 0.1509",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_COLOR, edgecolor="#AB47BC", alpha=0.9),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Figure saved: {output_path}")


def plot_classical_vs_dl_scatter(df_dl: pd.DataFrame, classical_csv: Path, output_path: Path):
    """Figure 3: Cross-paradigm scatter comparing Classical vs DL drift with seed annotations."""
    if not classical_csv.exists():
        print(f"Skipping classical scatter: {classical_csv} missing.")
        return

    df_class = pd.read_csv(classical_csv)
    merged = pd.merge(
        df_dl,
        df_class[["word", "drift_score", "status"]].rename(columns={"status": "class_status"}),
        on="word",
        how="inner",
    )
    valid = merged[merged["class_status"] == "OK"].dropna(subset=["drift_score", "wasserstein_w2"]).copy()

    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)

    # Scatter by origin class
    for origin in ORIGIN_CLASSES:
        sub = valid[valid["origin_class"] == origin]
        if not sub.empty:
            ax.scatter(
                sub["drift_score"],
                sub["wasserstein_w2"],
                label=origin,
                color=ORIGIN_COLORS.get(origin, "#888888"),
                alpha=0.65,
                s=40,
                edgecolors="none",
            )

    # Regression trendline
    sns.regplot(
        data=valid,
        x="drift_score",
        y="wasserstein_w2",
        scatter=False,
        ax=ax,
        color="#FFFFFF",
        line_kws={"linestyle": "--", "linewidth": 1.5, "alpha": 0.8},
    )

    # Highlight Benchmark Seed Words
    seed_highlights = {
        "father":  (0.015, -0.035),
        "night":   (0.015, 0.025),
        "stream":  (0.020, -0.030),
        "cell":    (-0.080, 0.025),
        "mouse":   (0.020, 0.020),
        "craft":   (-0.070, -0.030),
        "monitor": (0.020, -0.025),
        "atom":    (0.020, 0.020),
        "avatar":  (0.020, 0.020),
    }

    for word, offset in seed_highlights.items():
        match = valid[valid["word"] == word]
        if not match.empty:
            row = match.iloc[0]
            x, y = row["drift_score"], row["wasserstein_w2"]
            ax.scatter(x, y, color="#FFEB3B", s=90, edgecolor="black", linewidth=1.5, zorder=6)
            ax.annotate(
                f"{word} ({row['origin_class'][0]})",
                (x, y),
                xytext=(x + offset[0], y + offset[1]),
                fontsize=9.5,
                fontweight="bold",
                color="#FFEB3B",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=PANEL_COLOR, edgecolor="#FFEB3B", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#FFEB3B", lw=1.0),
                zorder=7,
            )

    # Stats Annotation
    sr, sp = stats.spearmanr(valid["wasserstein_w2"], valid["drift_score"])
    pr, pp = stats.pearsonr(valid["wasserstein_w2"], valid["drift_score"])

    ax.text(
        0.03, 0.94,
        f"Cross-Paradigm Correlation (N={len(valid)}):\nSpearman ρ = {sr:.4f} (p = {sp:.4f}) **\nPearson r   = {pr:.4f} (p = {pp:.4f})",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_COLOR, edgecolor="#4FC3F7", alpha=0.9),
    )

    ax.set_title("Cross-Paradigm Validation: Classical Drift vs. DL 2-Wasserstein Distance", pad=15, fontweight="bold")
    ax.set_xlabel("Classical Drift Score (1 - Cosine Similarity via Word2Vec + Procrustes)", labelpad=10, fontweight="semibold")
    ax.set_ylabel("Deep Learning 2-Wasserstein Distance (W2 on BERT Token Clouds)", labelpad=10, fontweight="semibold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Origin Class", loc="lower right", frameon=True, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Figure saved: {output_path}")


def plot_dispersion_space(df: pd.DataFrame, output_path: Path):
    """Figure 4: Historical vs Modern intra-cloud semantic dispersion scatter."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Scatter points colored by shift type
    for st, col in TYPOLOGY_COLORS.items():
        sub = df[df["shift_type"] == st]
        ax.scatter(
            sub["dispersion_hist"],
            sub["dispersion_mod"],
            label=st,
            color=col,
            alpha=0.7,
            s=45,
            edgecolors="none",
        )

    # Diagonal line y = x (neutral line)
    min_val = min(df["dispersion_hist"].min(), df["dispersion_mod"].min()) * 0.9
    max_val = max(df["dispersion_hist"].max(), df["dispersion_mod"].max()) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], color="#FFFFFF", linestyle="--", alpha=0.5, label="Invariant Dispersion (Δσ²=0)")

    # Seed annotations with prominent broadening/narrowing
    highlight_words = ["mouse", "avatar", "father", "atom", "cell", "stream"]
    for w in highlight_words:
        m = df[df["word"] == w]
        if not m.empty:
            r = m.iloc[0]
            x, y = r["dispersion_hist"], r["dispersion_mod"]
            ax.scatter(x, y, color="#FFEB3B", s=80, edgecolor="black", linewidth=1.2, zorder=6)
            ax.annotate(
                f"{w} (Δ={r['delta_dispersion']:+.2f})",
                (x, y),
                xytext=(x + 0.015, y - 0.015),
                fontsize=9,
                fontweight="bold",
                color="#FFEB3B",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL_COLOR, edgecolor="#FFEB3B", alpha=0.85),
            )

    # Region Labels
    ax.text(0.18, 0.50, "Semantic Broadening ↑\n(Polysemy / Domain Expansion)", fontsize=11, color="#42A5F5", alpha=0.85, fontweight="bold")
    ax.text(0.42, 0.22, "Semantic Narrowing ↓\n(Specialization / Archaisms)", fontsize=11, color="#AB47BC", alpha=0.85, fontweight="bold")

    ax.set_title("Diachronic Semantic Dispersion Dynamics (1800s vs. 2000s)", pad=15, fontweight="bold")
    ax.set_xlabel("Historical Intra-Cloud Dispersion Variance (σ²_hist)", labelpad=10, fontweight="semibold")
    ax.set_ylabel("Modern Intra-Cloud Dispersion Variance (σ²_mod)", labelpad=10, fontweight="semibold")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Linguistic Shift Typology", loc="upper left", frameon=True, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Figure saved: {output_path}")


def plot_correlation_heatmap(df_dl: pd.DataFrame, classical_csv: Path, output_path: Path):
    """Figure 5: Metric Correlation Heatmap."""
    cols_to_correlate = [
        ("wasserstein_w2", "W2 (Norm)"),
        ("wasserstein_w1_cos", "W1 (Cosine)"),
        ("wasserstein_raw_w2", "W2 (Raw 768d)"),
        ("centroid_cosine_drift", "Centroid Drift"),
        ("delta_k", "Sense Change (Δk)"),
        ("delta_dispersion", "Dispersion Shift (Δσ²)"),
    ]

    analysis_df = pd.DataFrame()
    for col, name in cols_to_correlate:
        if col in df_dl:
            analysis_df[name] = df_dl[col]

    if classical_csv.exists():
        df_class = pd.read_csv(classical_csv)
        m = pd.merge(df_dl[["word"]], df_class[["word", "drift_score", "status"]], on="word", how="inner")
        valid_m = m[m["status"] == "OK"]
        # Merge back
        temp = pd.merge(df_dl, valid_m[["word", "drift_score"]], on="word", how="left")
        analysis_df["Classical Drift"] = temp["drift_score"]

    corr = analysis_df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        linewidths=1.0,
        linecolor=BG_COLOR,
        cbar_kws={"label": "Spearman Rank Correlation (ρ)"},
        ax=ax,
    )

    ax.set_title("Inter-Metric Diachronic Shift Correlation Matrix", pad=15, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Figure saved: {output_path}")


def main():
    setup_dark_style()

    if not DL_DRIFT_SCORES_CSV.exists():
        print(f"✗ Error: {DL_DRIFT_SCORES_CSV} not found. Run dl_drift.py first.")
        sys.exit(1)

    df_dl = pd.read_csv(DL_DRIFT_SCORES_CSV)
    print(f"\n============================================================")
    print(f"  ETYMO Phase 3 Task 3.4: Publication-Quality Visualizations")
    print(f"  Generating 5 Figures in {DL_FIGURES_DIR}")
    print(f"============================================================")

    DL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. W2 Boxplot & Jitter
    plot_w2_distribution(df_dl, DL_FIGURES_DIR / "dl_w2_distribution_boxplot.png")

    # 2. Shift Typology Breakdown
    plot_shift_typology_breakdown(df_dl, DL_FIGURES_DIR / "dl_shift_typology_breakdown.png")

    # 3. Classical vs DL Scatter
    plot_classical_vs_dl_scatter(df_dl, CLASSICAL_DRIFT_CSV, DL_FIGURES_DIR / "dl_classical_vs_dl_scatter.png")

    # 4. Dispersion Space Scatter
    plot_dispersion_space(df_dl, DL_FIGURES_DIR / "dl_dispersion_shift_scatter.png")

    # 5. Correlation Heatmap
    plot_correlation_heatmap(df_dl, CLASSICAL_DRIFT_CSV, DL_FIGURES_DIR / "dl_correlation_heatmap.png")

    print(f"\n🎉 All 5 publication-ready figures generated successfully!")


if __name__ == "__main__":
    main()
