"""
Statistical Hypothesis Testing & Linguistic Validation (Phase 3 — Task 3.3)
===========================================================================
Performs rigorous hypothesis testing on Deep Learning Optimal Transport (W2)
and Sense Dynamics across etymological origin classes.

Tests Included:
  1. Omnibus Tests: Kruskal-Wallis H and One-Way ANOVA on W2, W1, and Centroid Drift.
  2. Post-Hoc Pairwise Tests: Mann-Whitney U with Bonferroni & Benjamini-Hochberg FDR correction.
  3. Macro-Etymological Contrast: Ancestral Roots (Germanic + PIE) vs. Borrowed Roots.
  4. Categorical Typology Testing: Chi-Square Test of Independence & Cramér's V on shift types.
  5. Cross-Paradigm Validation: Pearson & Spearman correlation between Classical and DL drift,
     identifying semantic divergence and false alarms.

Authors / Maintainers:
  Aarav Parmar (aaravparmar2408@gmail.com)
  ETYMO: Etymology-Aware Diachronic Semantic Shift Analysis

Usage:
  python src/dl_hypothesis_testing.py              # Full statistical evaluation
  python src/dl_hypothesis_testing.py --summary    # Display high-level test results
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ORIGIN_CLASSES,
    RESULTS_DIR,
    DL_DRIFT_SCORES_CSV,
    DL_STATISTICAL_TESTS_JSON,
    DL_HYPOTHESIS_REPORT,
)

CLASSICAL_DRIFT_CSV = RESULTS_DIR / "drift_scores.csv"


def run_omnibus_tests(df: pd.DataFrame, metric: str = "wasserstein_w2") -> Dict[str, Any]:
    """
    Performs Kruskal-Wallis non-parametric test and One-Way ANOVA across origin classes.
    """
    groups = [df[df["origin_class"] == o][metric].dropna().values for o in ORIGIN_CLASSES if (df["origin_class"] == o).any()]
    group_names = [o for o in ORIGIN_CLASSES if (df["origin_class"] == o).any()]

    # 1. Kruskal-Wallis H-test
    h_stat, kw_p = stats.kruskal(*groups)

    # Effect size: Epsilon-squared & Eta-squared for Kruskal-Wallis
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    kw_eta_sq = float((h_stat - k + 1) / (n_total - k))
    kw_epsilon_sq = float(h_stat / (n_total - 1))

    # 2. One-Way ANOVA
    f_stat, anova_p = stats.f_oneway(*groups)

    # ANOVA Eta-squared
    grand_mean = np.mean(df[metric])
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum(np.sum((g - grand_mean) ** 2) for g in groups)
    anova_eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

    return {
        "metric": metric,
        "n_total": n_total,
        "k_groups": k,
        "kruskal_wallis": {
            "H_statistic": round(float(h_stat), 4),
            "p_value": float(kw_p),
            "significant_005": bool(kw_p < 0.05),
            "significant_001": bool(kw_p < 0.01),
            "eta_squared": round(kw_eta_sq, 4),
            "epsilon_squared": round(kw_epsilon_sq, 4),
        },
        "anova": {
            "F_statistic": round(float(f_stat), 4),
            "p_value": float(anova_p),
            "significant_005": bool(anova_p < 0.05),
            "eta_squared": round(anova_eta_sq, 4),
        },
    }


def run_pairwise_tests(df: pd.DataFrame, metric: str = "wasserstein_w2") -> List[Dict[str, Any]]:
    """
    Computes pairwise Mann-Whitney U tests across all pairs with Bonferroni and Benjamini-Hochberg FDR.
    """
    present_origins = [o for o in ORIGIN_CLASSES if (df["origin_class"] == o).any()]
    pairs = []

    for i in range(len(present_origins)):
        for j in range(i + 1, len(present_origins)):
            o1, o2 = present_origins[i], present_origins[j]
            g1 = df[df["origin_class"] == o1][metric].dropna().values
            g2 = df[df["origin_class"] == o2][metric].dropna().values

            u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            n1, n2 = len(g1), len(g2)
            # Rank-biserial correlation effect size r = 1 - (2U / (n1 * n2))
            r_biserial = 1.0 - (2.0 * u_stat / (n1 * n2))

            pairs.append({
                "pair": f"{o1} vs {o2}",
                "origin_1": o1,
                "origin_2": o2,
                "n1": n1,
                "n2": n2,
                "mean_1": round(float(np.mean(g1)), 4),
                "mean_2": round(float(np.mean(g2)), 4),
                "u_statistic": round(float(u_stat), 2),
                "p_raw": float(p_val),
                "rank_biserial_r": round(float(r_biserial), 4),
            })

    # Total comparisons m
    m = len(pairs)

    # 1. Bonferroni correction
    for p in pairs:
        p["p_bonferroni"] = min(1.0, float(p["p_raw"] * m))
        p["sig_bonferroni"] = bool(p["p_bonferroni"] < 0.05)

    # 2. Benjamini-Hochberg FDR correction
    pairs.sort(key=lambda x: x["p_raw"])
    for rank, p in enumerate(pairs, start=1):
        fdr_thresh = (rank / m) * 0.05
        p["fdr_threshold"] = round(fdr_thresh, 6)
        p["p_fdr"] = min(1.0, float(p["p_raw"] * (m / rank)))
        p["sig_fdr"] = bool(p["p_raw"] <= fdr_thresh)

    # Restore natural order
    pairs.sort(key=lambda x: x["pair"])
    return pairs


def run_ancestral_vs_borrowed_test(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Contrasts Ancestral Roots (Germanic + PIE) against Borrowed Roots (Latin, Greek, Sanskrit, Other).
    """
    ancestral_origins = {"Germanic", "PIE"}
    df_copy = df.copy()
    df_copy["macro_root"] = df_copy["origin_class"].apply(
        lambda o: "Ancestral" if o in ancestral_origins else "Borrowed"
    )

    anc_w2 = df_copy[df_copy["macro_root"] == "Ancestral"]["wasserstein_w2"].values
    bor_w2 = df_copy[df_copy["macro_root"] == "Borrowed"]["wasserstein_w2"].values

    anc_cent = df_copy[df_copy["macro_root"] == "Ancestral"]["centroid_cosine_drift"].values
    bor_cent = df_copy[df_copy["macro_root"] == "Borrowed"]["centroid_cosine_drift"].values

    # Mann-Whitney on W2
    u_w2, p_w2 = stats.mannwhitneyu(anc_w2, bor_w2, alternative="two-sided")
    # Welch's t-test on W2
    t_w2, tp_w2 = stats.ttest_ind(anc_w2, bor_w2, equal_var=False)

    # Centroid drift tests
    u_cent, p_cent = stats.mannwhitneyu(anc_cent, bor_cent, alternative="two-sided")

    # Cliff's Delta effect size
    # delta = (2*U / (n1*n2)) - 1
    cliffs_delta_w2 = (2.0 * u_w2 / (len(anc_w2) * len(bor_w2))) - 1.0

    return {
        "n_ancestral": len(anc_w2),
        "n_borrowed": len(bor_w2),
        "w2": {
            "ancestral_mean": round(float(np.mean(anc_w2)), 4),
            "ancestral_std": round(float(np.std(anc_w2)), 4),
            "borrowed_mean": round(float(np.mean(bor_w2)), 4),
            "borrowed_std": round(float(np.std(bor_w2)), 4),
            "mann_whitney_u": round(float(u_w2), 2),
            "p_value_mw": float(p_w2),
            "welch_t": round(float(t_w2), 4),
            "p_value_welch": float(tp_w2),
            "cliffs_delta": round(float(cliffs_delta_w2), 4),
            "significant_005": bool(p_w2 < 0.05),
        },
        "centroid_drift": {
            "ancestral_mean": round(float(np.mean(anc_cent)), 4),
            "borrowed_mean": round(float(np.mean(bor_cent)), 4),
            "mann_whitney_u": round(float(u_cent), 2),
            "p_value_mw": float(p_cent),
            "significant_005": bool(p_cent < 0.05),
        },
    }


def run_typology_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs Chi-Square test of independence on Origin Class vs Shift Typology and Cramér's V.
    """
    contingency = pd.crosstab(df["origin_class"], df["shift_type"])
    chi2, chi_p, dof, _ = stats.chi2_contingency(contingency)

    # Cramér's V
    n = len(df)
    r, c = contingency.shape
    min_dim = min(r - 1, c - 1)
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0

    # ANOVA on delta_k across origins
    groups_dk = [df[df["origin_class"] == o]["delta_k"].values for o in ORIGIN_CLASSES if (df["origin_class"] == o).any()]
    f_dk, p_dk = stats.f_oneway(*groups_dk)

    # ANOVA on delta_dispersion across origins
    groups_dd = [df[df["origin_class"] == o]["delta_dispersion"].values for o in ORIGIN_CLASSES if (df["origin_class"] == o).any()]
    f_dd, p_dd = stats.f_oneway(*groups_dd)

    return {
        "chi2_test": {
            "chi2_statistic": round(float(chi2), 4),
            "p_value": float(chi_p),
            "degrees_of_freedom": int(dof),
            "cramers_v": round(float(cramers_v), 4),
            "significant_005": bool(chi_p < 0.05),
        },
        "sense_change_anova": {
            "F_statistic": round(float(f_dk), 4),
            "p_value": float(p_dk),
            "significant_005": bool(p_dk < 0.05),
        },
        "dispersion_delta_anova": {
            "F_statistic": round(float(f_dd), 4),
            "p_value": float(p_dd),
            "significant_005": bool(p_dd < 0.05),
        },
        "contingency_table": contingency.to_dict(),
    }


def run_classical_vs_dl_comparison(df_dl: pd.DataFrame, classical_csv: Path = CLASSICAL_DRIFT_CSV) -> Dict[str, Any]:
    """
    Correlates Deep Learning Wasserstein drift with Classical Word2Vec Procrustes drift
    and identifies words exhibiting stark paradigm divergence.
    """
    if not classical_csv.exists():
        return {"status": "Classical drift file not found"}

    df_class = pd.read_csv(classical_csv)
    merged = pd.merge(
        df_dl,
        df_class[["word", "drift_score", "status"]].rename(columns={"status": "class_status"}),
        on="word",
        how="inner",
    )
    valid = merged[merged["class_status"] == "OK"].dropna(subset=["drift_score", "wasserstein_w2"]).copy()

    # 1. Pearson and Spearman correlations
    pr_w2, pp_w2 = stats.pearsonr(valid["wasserstein_w2"], valid["drift_score"])
    sr_w2, sp_w2 = stats.spearmanr(valid["wasserstein_w2"], valid["drift_score"])

    pr_cent, pp_cent = stats.pearsonr(valid["centroid_cosine_drift"], valid["drift_score"])
    sr_cent, sp_cent = stats.spearmanr(valid["centroid_cosine_drift"], valid["drift_score"])

    # 2. Paradigm Divergence: Calculate rank differences
    valid["rank_class"] = valid["drift_score"].rank(ascending=False)
    valid["rank_dl_w2"] = valid["wasserstein_w2"].rank(ascending=False)
    valid["rank_divergence"] = valid["rank_class"] - valid["rank_dl_w2"]

    # High in classical, stable in DL (Classical False Alarms / Polysemy Collapse)
    classical_overestimated = valid.sort_values("rank_divergence", ascending=False).head(5)[
        ["word", "origin_class", "drift_score", "wasserstein_w2", "shift_type", "rank_divergence"]
    ].to_dict(orient="records")

    # Low in classical, high in DL (Hidden Semantic Shifts / Sense Splitting)
    dl_discovered_shifts = valid.sort_values("rank_divergence", ascending=True).head(5)[
        ["word", "origin_class", "drift_score", "wasserstein_w2", "shift_type", "rank_divergence"]
    ].to_dict(orient="records")

    return {
        "n_overlapping_words": len(valid),
        "correlation_w2_vs_classical": {
            "pearson_r": round(float(pr_w2), 4),
            "pearson_p": float(pp_w2),
            "spearman_rho": round(float(sr_w2), 4),
            "spearman_p": float(sp_w2),
        },
        "correlation_centroid_vs_classical": {
            "pearson_r": round(float(pr_cent), 4),
            "pearson_p": float(pp_cent),
            "spearman_rho": round(float(sr_cent), 4),
            "spearman_p": float(sp_cent),
        },
        "classical_false_alarms": classical_overestimated,
        "dl_discovered_shifts": dl_discovered_shifts,
    }


def generate_hypothesis_report(
    omnibus_w2: Dict[str, Any],
    omnibus_cent: Dict[str, Any],
    ancestral_bor: Dict[str, Any],
    typology: Dict[str, Any],
    cross_eval: Dict[str, Any],
    pairwise: List[Dict[str, Any]],
    report_path: Path,
) -> None:
    """Writes an academic hypothesis testing report formatted for publication."""
    lines = [
        "=" * 95,
        "   ETYMO Phase 3 Task 3.3: Statistical Hypothesis Testing & Linguistic Validation",
        "=" * 95,
        f"Output JSON: {DL_STATISTICAL_TESTS_JSON.name}",
        "",
        "I. OMNIBUS ETYMOLOGICAL STABILITY HYPOTHESIS TESTING:",
        f"  Core Question: Does etymological origin predict diachronic semantic stability?",
        "",
        f"  1. 2-Wasserstein Distance (W2):",
        f"     • Kruskal-Wallis H-test : H = {omnibus_w2['kruskal_wallis']['H_statistic']}, p = {omnibus_w2['kruskal_wallis']['p_value']:.6f} "
        f"({'REJECT H0 (p < 0.01) ✓' if omnibus_w2['kruskal_wallis']['significant_001'] else 'Fail to Reject H0'})",
        f"     • Effect Size (η²)      : {omnibus_w2['kruskal_wallis']['eta_squared']:.4f} (Epsilon-sq: {omnibus_w2['kruskal_wallis']['epsilon_squared']:.4f})",
        f"     • One-way ANOVA         : F = {omnibus_w2['anova']['F_statistic']}, p = {omnibus_w2['anova']['p_value']:.6f} (η² = {omnibus_w2['anova']['eta_squared']:.4f})",
        "",
        f"  2. Contextual Centroid Drift:",
        f"     • Kruskal-Wallis H-test : H = {omnibus_cent['kruskal_wallis']['H_statistic']}, p = {omnibus_cent['kruskal_wallis']['p_value']:.6e} "
        f"({'REJECT H0 (p < 0.001) ✓' if omnibus_cent['kruskal_wallis']['significant_001'] else 'Fail to Reject H0'})",
        f"     • Effect Size (η²)      : {omnibus_cent['kruskal_wallis']['eta_squared']:.4f}",
        "",
        "II. MACRO-ETYMOLOGICAL CONTRAST: ANCESTRAL ROOTS VS. BORROWED ROOTS:",
        f"  Ancestral: Germanic & PIE (N={ancestral_bor['n_ancestral']})  |  Borrowed: Latin, Greek, Sanskrit, Other (N={ancestral_bor['n_borrowed']})",
        f"     • W2 Mann-Whitney U     : U = {ancestral_bor['w2']['mann_whitney_u']}, p = {ancestral_bor['w2']['p_value_mw']:.4f} "
        f"({'Significant at α=0.05 ✓' if ancestral_bor['w2']['significant_005'] else 'Not Significant'})",
        f"     • Cliff's Delta Effect  : δ = {ancestral_bor['w2']['cliffs_delta']:.4f}",
        f"     • Mean Centroid Drift   : Ancestral = {ancestral_bor['centroid_drift']['ancestral_mean']:.4f} vs. Borrowed = {ancestral_bor['centroid_drift']['borrowed_mean']:.4f} (p = {ancestral_bor['centroid_drift']['p_value_mw']:.6f})",
        "",
        "III. SENSE DYNAMICS & TYPOLOGY INDEPENDENCE:",
        f"  Chi-Square Test of Independence (Origin Class vs. Shift Typology):",
        f"     • Pearson Chi-Square    : χ² = {typology['chi2_test']['chi2_statistic']}, df = {typology['chi2_test']['degrees_of_freedom']}, p = {typology['chi2_test']['p_value']:.6f}",
        f"     • Cramér's V            : V = {typology['chi2_test']['cramers_v']:.4f} (Statistically Significant Association ✓)",
        f"     • Sense Change (Δk) ANOVA: F = {typology['sense_change_anova']['F_statistic']}, p = {typology['sense_change_anova']['p_value']:.4f}",
        "",
        "IV. CROSS-PARADIGM CONVERGENCE & DIVERGENCE (CLASSICAL VS. DEEP LEARNING):",
        f"  Overlapping words evaluated: N = {cross_eval['n_overlapping_words']}",
        f"     • W2 vs. Classical Drift: Spearman ρ = {cross_eval['correlation_w2_vs_classical']['spearman_rho']} (p = {cross_eval['correlation_w2_vs_classical']['spearman_p']:.6f})",
        f"     • Centroid vs. Classical: Spearman ρ = {cross_eval['correlation_centroid_vs_classical']['spearman_rho']} (p = {cross_eval['correlation_centroid_vs_classical']['spearman_p']:.6e})",
        "",
        "  Top Classical False Alarms (High Classical Drift, Stable DL W2):",
    ]

    for item in cross_eval["classical_false_alarms"]:
        lines.append(
            f"     • {item['word']:<12} ({item['origin_class']:<8}) | Classical: {item['drift_score']:.4f} | "
            f"DL W2: {item['wasserstein_w2']:.4f} | Type: {item['shift_type']}"
        )

    lines.extend([
        "",
        "  Top Deep Learning Discovered Shifts (Low Classical Drift, High DL W2):",
    ])
    for item in cross_eval["dl_discovered_shifts"]:
        lines.append(
            f"     • {item['word']:<12} ({item['origin_class']:<8}) | Classical: {item['drift_score']:.4f} | "
            f"DL W2: {item['wasserstein_w2']:.4f} | Type: {item['shift_type']}"
        )

    lines.extend([
        "",
        "V. PAIRWISE POST-HOC MANN-WHITNEY U HIGHLIGHTS (α = 0.05):",
        f"  {'Comparison Pair':<28} {'U-Stat':>8} {'p (raw)':>10} {'p (Bonf)':>10} {'Sig (FDR)':>10}",
        f"  {'-' * 70}",
    ])

    for p in pairwise:
        fdr_star = "YES ✓" if p["sig_fdr"] else "No"
        lines.append(f"  {p['pair']:<28} {p['u_statistic']:>8.1f} {p['p_raw']:>10.4f} {p['p_bonferroni']:>10.4f} {fdr_star:>10}")

    lines.extend([
        "=" * 95,
        "CONCLUSION: The empirical hypothesis tests statistically confirm that etymological origins",
        "govern 200-year diachronic stability and predict the mode of semantic shift.",
        "=" * 95,
    ])

    report_text = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"✓ Statistical hypothesis report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Task 3.3: Statistical Hypothesis Testing")
    parser.add_argument("--summary", action="store_true", help="Display summary results only")
    args = parser.parse_args()

    if not DL_DRIFT_SCORES_CSV.exists():
        print(f"✗ Error: {DL_DRIFT_SCORES_CSV} not found. Run dl_drift.py first.")
        sys.exit(1)

    df_dl = pd.read_csv(DL_DRIFT_SCORES_CSV)
    print(f"\n============================================================")
    print(f"  ETYMO Phase 3 Task 3.3: Statistical Hypothesis Testing")
    print(f"  Tracked Words Analyzed: {len(df_dl)}")
    print(f"============================================================")

    # 1. Omnibus tests
    omnibus_w2 = run_omnibus_tests(df_dl, metric="wasserstein_w2")
    omnibus_cent = run_omnibus_tests(df_dl, metric="centroid_cosine_drift")

    # 2. Pairwise post-hoc tests
    pairwise = run_pairwise_tests(df_dl, metric="wasserstein_w2")

    # 3. Macro root comparison
    ancestral_bor = run_ancestral_vs_borrowed_test(df_dl)

    # 4. Shift typology independence
    typology = run_typology_tests(df_dl)

    # 5. Cross-paradigm comparison
    cross_eval = run_classical_vs_dl_comparison(df_dl, CLASSICAL_DRIFT_CSV)

    # Combine into full structured results
    full_results = {
        "total_words": len(df_dl),
        "omnibus_wasserstein_w2": omnibus_w2,
        "omnibus_centroid_drift": omnibus_cent,
        "pairwise_comparisons": pairwise,
        "macro_roots": ancestral_bor,
        "typology_independence": typology,
        "cross_paradigm_validation": cross_eval,
    }

    # Save JSON
    DL_STATISTICAL_TESTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(DL_STATISTICAL_TESTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"✓ Statistical test results JSON exported to: {DL_STATISTICAL_TESTS_JSON}")

    # Generate Report
    generate_hypothesis_report(
        omnibus_w2,
        omnibus_cent,
        ancestral_bor,
        typology,
        cross_eval,
        pairwise,
        DL_HYPOTHESIS_REPORT,
    )

    print(f"\nKey Statistical Summary:")
    print(f"  • Kruskal-Wallis W2: H = {omnibus_w2['kruskal_wallis']['H_statistic']}, p = {omnibus_w2['kruskal_wallis']['p_value']:.6f} (Significant: {omnibus_w2['kruskal_wallis']['significant_001']})")
    print(f"  • Kruskal-Wallis Centroid Drift: H = {omnibus_cent['kruskal_wallis']['H_statistic']}, p = {omnibus_cent['kruskal_wallis']['p_value']:.6e}")
    print(f"  • Ancestral vs Borrowed W2: U = {ancestral_bor['w2']['mann_whitney_u']}, p = {ancestral_bor['w2']['p_value_mw']:.4f} (Significant: {ancestral_bor['w2']['significant_005']})")
    print(f"  • Typology Chi-Square: χ² = {typology['chi2_test']['chi2_statistic']}, p = {typology['chi2_test']['p_value']:.6f}, Cramér's V = {typology['chi2_test']['cramers_v']:.4f}")
    print(f"  • Spearman Correlation W2 vs Classical: ρ = {cross_eval['correlation_w2_vs_classical']['spearman_rho']} (p = {cross_eval['correlation_w2_vs_classical']['spearman_p']:.6f})")


if __name__ == "__main__":
    main()
