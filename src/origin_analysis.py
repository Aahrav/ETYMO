"""
Origin-Wise Analysis (Phase 8 — THE NOVELTY)
===============================================
Aggregates drift by etymology class and tests whether etymological
origin predicts semantic stability — the core research contribution.

Steps:
  8.1  Aggregate per-origin statistics (mean, median, std, min/max)
  8.2  Statistical significance testing (Kruskal-Wallis + pairwise Mann-Whitney)
  8.3  Generate interpretation report

Usage:
    python src/origin_analysis.py
    python src/origin_analysis.py --summary   # Quick summary
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS_DIR, ORIGIN_CLASSES

DRIFT_SCORES_CSV = RESULTS_DIR / "drift_scores.csv"
ORIGIN_SUMMARY_CSV = RESULTS_DIR / "origin_drift_summary.csv"
ORIGIN_REPORT = RESULTS_DIR / "origin_analysis_report.txt"


def main():
    parser = argparse.ArgumentParser(description="Phase 8: Origin-wise drift analysis")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if not DRIFT_SCORES_CSV.exists():
        print("✗ Run compute_drift.py first")
        sys.exit(1)

    df = pd.read_csv(DRIFT_SCORES_CSV)
    valid = df[df["status"] == "OK"].copy()

    print("=" * 60)
    print("  Phase 8: Origin-Wise Drift Analysis")
    print("  Core Question: Do etymological origins predict stability?")
    print("=" * 60)

    # ── 8.1: Aggregate statistics ──
    print("\n[Step 8.1] Aggregating per-origin statistics...")

    summary_records = []
    origin_groups = {}

    for origin in ORIGIN_CLASSES:
        og = valid[valid["origin_class"] == origin]["drift_score"]
        origin_groups[origin] = og.values

        rec = {
            "origin_class": origin,
            "n_words": len(og),
            "mean_drift": round(og.mean(), 4),
            "median_drift": round(og.median(), 4),
            "std_drift": round(og.std(), 4),
            "min_drift": round(og.min(), 4),
            "max_drift": round(og.max(), 4),
            "min_word": valid[valid["origin_class"] == origin].nsmallest(1, "drift_score")["word"].iloc[0],
            "max_word": valid[valid["origin_class"] == origin].nlargest(1, "drift_score")["word"].iloc[0],
        }
        summary_records.append(rec)

        print(f"\n  {origin} ({rec['n_words']} words):")
        print(f"    Mean:   {rec['mean_drift']:.4f}")
        print(f"    Median: {rec['median_drift']:.4f}")
        print(f"    Std:    {rec['std_drift']:.4f}")
        print(f"    Range:  [{rec['min_drift']:.4f}, {rec['max_drift']:.4f}]")
        print(f"    Most stable:  {rec['min_word']} ({rec['min_drift']:.4f})")
        print(f"    Most drifted: {rec['max_word']} ({rec['max_drift']:.4f})")

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(ORIGIN_SUMMARY_CSV, index=False)
    print(f"\n  ✓ Saved to {ORIGIN_SUMMARY_CSV.name}")

    # ── 8.2: Statistical testing ──
    print(f"\n[Step 8.2] Statistical significance testing...")

    # Kruskal-Wallis H-test (non-parametric alternative to ANOVA)
    # Better than ANOVA because we can't assume normal distributions
    groups = [origin_groups[o] for o in ORIGIN_CLASSES if len(origin_groups[o]) > 0]
    group_labels = [o for o in ORIGIN_CLASSES if len(origin_groups[o]) > 0]

    h_stat, kw_p = scipy_stats.kruskal(*groups)
    print(f"\n  Kruskal-Wallis H-test:")
    print(f"    H-statistic:  {h_stat:.4f}")
    print(f"    p-value:      {kw_p:.6f}")
    sig = kw_p < 0.05
    print(f"    Significant:  {'✓ YES (p < 0.05)' if sig else '✗ NO (p ≥ 0.05)'}")

    # Also run one-way ANOVA for comparison
    f_stat, anova_p = scipy_stats.f_oneway(*groups)
    print(f"\n  One-way ANOVA (for comparison):")
    print(f"    F-statistic:  {f_stat:.4f}")
    print(f"    p-value:      {anova_p:.6f}")

    # Pairwise Mann-Whitney U tests (if Kruskal-Wallis is significant)
    pairwise_results = []
    print(f"\n  Pairwise Mann-Whitney U tests:")
    print(f"  {'Pair':<25} {'U':>8} {'p-value':>10} {'Significant':>12}")
    print(f"  {'─' * 57}")

    for i in range(len(group_labels)):
        for j in range(i + 1, len(group_labels)):
            label_a = group_labels[i]
            label_b = group_labels[j]
            u_stat, mw_p = scipy_stats.mannwhitneyu(
                origin_groups[label_a],
                origin_groups[label_b],
                alternative="two-sided",
            )
            pair_sig = mw_p < 0.05
            pair_label = f"{label_a} vs {label_b}"
            pairwise_results.append({
                "pair": pair_label,
                "U_stat": round(u_stat, 2),
                "p_value": round(mw_p, 6),
                "significant": pair_sig,
            })
            star = "✓ *" if pair_sig else "  "
            print(f"  {pair_label:<25} {u_stat:>8.1f} {mw_p:>10.6f} {star:>12}")

    # Effect size (eta-squared for Kruskal-Wallis)
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    eta_sq = (h_stat - k + 1) / (n_total - k)
    effect_label = "small" if eta_sq < 0.06 else "medium" if eta_sq < 0.14 else "large"
    print(f"\n  Effect size (η²): {eta_sq:.4f} ({effect_label})")

    # ── 8.3: Interpretation ──
    print(f"\n[Step 8.3] Interpretation...")

    sorted_origins = sorted(summary_records, key=lambda x: x["mean_drift"])
    most_stable = sorted_origins[0]
    most_drifted = sorted_origins[-1]

    interpretation = []
    interpretation.append(f"\n  Key Finding:")
    interpretation.append(f"    Most stable origin:  {most_stable['origin_class']} "
                         f"(mean drift = {most_stable['mean_drift']:.4f})")
    interpretation.append(f"    Most drifted origin: {most_drifted['origin_class']} "
                         f"(mean drift = {most_drifted['mean_drift']:.4f})")

    if sig:
        interpretation.append(f"\n    ✓ The difference IS statistically significant (p = {kw_p:.4f})")
        interpretation.append(f"    → Etymological origin appears to predict semantic stability")
    else:
        interpretation.append(f"\n    ✗ The difference is NOT statistically significant (p = {kw_p:.4f})")
        interpretation.append(f"    → Etymological origin alone does not predict semantic stability")

    for line in interpretation:
        print(line)

    # ── Write comprehensive report ──
    lines = []
    lines.append("=" * 70)
    lines.append("  Phase 8: Origin-Wise Drift Analysis Report")
    lines.append("  Core Question: Do etymological origins predict semantic stability?")
    lines.append("=" * 70)
    lines.append("")

    lines.append("  Per-Origin Summary:")
    lines.append(f"  {'Origin':<12} {'N':>4} {'Mean':>8} {'Median':>8} "
                f"{'Std':>8} {'Min':>8} {'Max':>8} {'Most Stable':<15} {'Most Drifted':<15}")
    lines.append(f"  {'─' * 95}")
    for r in summary_records:
        lines.append(f"  {r['origin_class']:<12} {r['n_words']:>4} "
                    f"{r['mean_drift']:>8.4f} {r['median_drift']:>8.4f} "
                    f"{r['std_drift']:>8.4f} {r['min_drift']:>8.4f} "
                    f"{r['max_drift']:>8.4f} {r['min_word']:<15} {r['max_word']:<15}")

    lines.append("")
    lines.append(f"  Kruskal-Wallis H-test: H={h_stat:.4f}, p={kw_p:.6f}")
    lines.append(f"  One-way ANOVA:        F={f_stat:.4f}, p={anova_p:.6f}")
    lines.append(f"  Effect size (η²):     {eta_sq:.4f} ({effect_label})")
    lines.append("")
    lines.append(f"  Significant: {'YES' if sig else 'NO'} (α = 0.05)")
    lines.append("")

    lines.append("  Pairwise Mann-Whitney U Tests:")
    for pr in pairwise_results:
        star = " *" if pr["significant"] else ""
        lines.append(f"    {pr['pair']:<25} U={pr['U_stat']:<8} p={pr['p_value']:.6f}{star}")

    lines.append("")
    for line in interpretation:
        lines.append(line)

    report = "\n".join(lines)
    ORIGIN_REPORT.write_text(report, encoding="utf-8")
    print(f"\n  ✓ Report saved to {ORIGIN_REPORT.name}")

    if args.summary:
        return

    print(f"\n{'=' * 60}")
    print(f"  Phase 8 Complete!")
    print(f"  Next: python src/visualize_static.py (Phase 9)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
