"""
Semantic Drift Computation (Phase 7)
======================================
Measures how much each tracked word's meaning shifted between 1800s and 2000s.

Steps:
  7.1  Drift scores: drift = 1 - cosine_similarity(aligned_old, new)
  7.2  Neighbor analysis: top-5 nearest neighbors in each period
  7.3  UMAP 2D projection: project aligned embeddings for visualization

Output:
  results/drift_scores.csv     — word, origin_class, drift_score, similarity
  results/neighbor_analysis.txt — human-readable neighbor comparison
  results/umap_coords.csv      — word, origin, x_old, y_old, x_new, y_new

Usage:
    python src/compute_drift.py             # Full computation
    python src/compute_drift.py --show      # Display drift rankings
    python src/compute_drift.py --neighbors WORD  # Show one word's neighbors
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    EMBEDDINGS_DIR, W2V_MODEL_OLD, W2V_MODEL_NEW, W2V_ALIGNED_OLD,
    RESULTS_DIR, SELECTED_WORDS_CSV, ORIGIN_CLASSES,
    TOP_N_NEIGHBORS, RANDOM_SEED, W2V_VECTOR_SIZE,
)

DRIFT_SCORES_CSV = RESULTS_DIR / "drift_scores.csv"
NEIGHBOR_REPORT = RESULTS_DIR / "neighbor_analysis.txt"
UMAP_COORDS_CSV = RESULTS_DIR / "umap_coords.csv"


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────
def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def load_aligned_data():
    """Load models, aligned vectors, and vocabulary."""
    print("  Loading models and aligned vectors...")

    model_old = Word2Vec.load(str(W2V_MODEL_OLD))
    model_new = Word2Vec.load(str(W2V_MODEL_NEW))
    aligned_old = np.load(str(W2V_ALIGNED_OLD))

    # Build word→index mapping for aligned old vectors
    old_words = list(model_old.wv.key_to_index.keys())
    old_w2i = {w: i for i, w in enumerate(old_words)}

    print(f"    Historical: {len(model_old.wv):,} words")
    print(f"    Modern:     {len(model_new.wv):,} words")
    print(f"    Aligned:    {aligned_old.shape}")

    return model_old, model_new, aligned_old, old_words, old_w2i


# ──────────────────────────────────────────────────────────────
#  Step 7.1: Compute drift scores
# ──────────────────────────────────────────────────────────────
def compute_drift_scores(
    model_new: Word2Vec,
    aligned_old: np.ndarray,
    old_w2i: dict,
) -> pd.DataFrame:
    """
    Compute semantic drift for all selected words.
    drift = 1 - cosine_similarity(aligned_old_vec, new_vec)
    """
    print("\n[Step 7.1] Computing drift scores...")

    selected = pd.read_csv(SELECTED_WORDS_CSV)
    records = []

    for _, row in selected.iterrows():
        word = row["word"]
        origin = row["origin_class"]

        in_old = word in old_w2i
        in_new = word in model_new.wv

        if in_old and in_new:
            vec_old = aligned_old[old_w2i[word]]
            vec_new = model_new.wv[word]
            sim = cosine_sim(vec_old, vec_new)
            drift = 1.0 - sim
            status = "OK"
        else:
            sim = float("nan")
            drift = float("nan")
            status = "MISSING"
            if not in_old:
                status += " (no historical)"
            if not in_new:
                status += " (no modern)"

        records.append({
            "word": word,
            "origin_class": origin,
            "drift_score": round(drift, 6) if not np.isnan(drift) else None,
            "similarity": round(sim, 6) if not np.isnan(sim) else None,
            "confidence": row.get("confidence", None),
            "source": row.get("source", None),
            "status": status,
        })

    df = pd.DataFrame(records)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DRIFT_SCORES_CSV, index=False)

    # Summary
    valid = df[df["status"] == "OK"]
    print(f"\n  ✓ Drift scores computed for {len(valid)}/{len(df)} words")
    print(f"    Mean drift:   {valid['drift_score'].mean():.4f}")
    print(f"    Median drift: {valid['drift_score'].median():.4f}")
    print(f"    Std drift:    {valid['drift_score'].std():.4f}")
    print(f"    Range:        [{valid['drift_score'].min():.4f}, "
          f"{valid['drift_score'].max():.4f}]")

    # Top 10 highest drift
    top = valid.nlargest(10, "drift_score")
    print(f"\n  Top 10 most-drifted words:")
    print(f"  {'Word':<18} {'Origin':<12} {'Drift':>8}")
    print(f"  {'─' * 40}")
    for _, r in top.iterrows():
        bar = "█" * int(r["drift_score"] * 30)
        print(f"  {r['word']:<18} {r['origin_class']:<12} {r['drift_score']:>8.4f}  {bar}")

    # Top 10 lowest drift
    bottom = valid.nsmallest(10, "drift_score")
    print(f"\n  Top 10 most-stable words:")
    print(f"  {'Word':<18} {'Origin':<12} {'Drift':>8}")
    print(f"  {'─' * 40}")
    for _, r in bottom.iterrows():
        print(f"  {r['word']:<18} {r['origin_class']:<12} {r['drift_score']:>8.4f}")

    # Per-origin summary
    print(f"\n  Per-origin drift summary:")
    for origin in ORIGIN_CLASSES:
        og = valid[valid["origin_class"] == origin]
        if len(og) > 0:
            print(f"    {origin:<12} mean={og['drift_score'].mean():.4f} "
                  f"median={og['drift_score'].median():.4f} "
                  f"std={og['drift_score'].std():.4f} "
                  f"n={len(og)}")

    print(f"\n  ✓ Saved to {DRIFT_SCORES_CSV.name}")
    return df


# ──────────────────────────────────────────────────────────────
#  Step 7.2: Nearest neighbor analysis
# ──────────────────────────────────────────────────────────────
def neighbor_analysis(
    model_old: Word2Vec,
    model_new: Word2Vec,
    aligned_old: np.ndarray,
    old_words: list,
    old_w2i: dict,
    drift_df: pd.DataFrame,
):
    """
    For each selected word, show its top-N nearest neighbors in
    both the historical (aligned) and modern spaces.
    """
    print("\n[Step 7.2] Nearest neighbor analysis...")

    valid = drift_df[drift_df["status"] == "OK"].sort_values(
        "drift_score", ascending=False
    )

    lines = []
    lines.append("=" * 70)
    lines.append("  Phase 7: Nearest Neighbor Analysis")
    lines.append("  Top-5 neighbors per word per time period")
    lines.append("=" * 70)

    # Precompute aligned old norms for neighbor search
    old_norms = np.linalg.norm(aligned_old, axis=1)
    old_norms[old_norms == 0] = 1

    def get_aligned_old_neighbors(word, topn=TOP_N_NEIGHBORS):
        """Find nearest neighbors in the ALIGNED old space."""
        if word not in old_w2i:
            return []
        target = aligned_old[old_w2i[word]]
        target_norm = np.linalg.norm(target)
        if target_norm == 0:
            return []
        sims = aligned_old @ target / (old_norms * target_norm)
        top_idxs = np.argsort(sims)[::-1][1:topn + 1]  # Skip self
        return [(old_words[i], float(sims[i])) for i in top_idxs]

    for _, row in valid.iterrows():
        word = row["word"]
        origin = row["origin_class"]
        drift = row["drift_score"]

        lines.append("")
        lines.append(f"{'─' * 70}")
        lines.append(f"  \"{word}\" ({origin})  │  drift = {drift:.4f}")
        lines.append(f"{'─' * 70}")

        # Old neighbors
        old_nbrs = get_aligned_old_neighbors(word)
        new_nbrs = model_new.wv.most_similar(word, topn=TOP_N_NEIGHBORS) \
            if word in model_new.wv else []

        lines.append(f"    1800s neighbors: {', '.join(w for w, _ in old_nbrs)}")
        lines.append(f"    2000s neighbors: {', '.join(w for w, _ in new_nbrs)}")

        # Check for neighbor overlap (stability indicator)
        old_set = {w for w, _ in old_nbrs}
        new_set = {w for w, _ in new_nbrs}
        overlap = old_set & new_set
        if overlap:
            lines.append(f"    Shared neighbors: {', '.join(overlap)}")
        else:
            lines.append(f"    Shared neighbors: none (complete context change)")

    report = "\n".join(lines)
    NEIGHBOR_REPORT.write_text(report, encoding="utf-8")
    print(f"  ✓ Saved to {NEIGHBOR_REPORT.name}")
    print(f"    {len(valid)} words analyzed with {TOP_N_NEIGHBORS} neighbors each")


# ──────────────────────────────────────────────────────────────
#  Step 7.3: UMAP 2D projection
# ──────────────────────────────────────────────────────────────
def compute_umap_projection(
    model_new: Word2Vec,
    aligned_old: np.ndarray,
    old_w2i: dict,
    drift_df: pd.DataFrame,
):
    """
    Project aligned embeddings to 2D using UMAP for visualization.

    Why UMAP:
      - Preserves local neighborhood structure (clusters stay tight)
      - Preserves global topology (distant clusters remain distant)
      - Deterministic with fixed seed
      - Much faster than t-SNE

    Process:
      1. Stack old + new vectors into one matrix (2N vectors)
      2. Fit UMAP on the combined matrix
      3. Split back into old_coords and new_coords
    """
    print("\n[Step 7.3] Computing UMAP 2D projection...")

    import umap

    valid = drift_df[drift_df["status"] == "OK"]
    words = valid["word"].tolist()
    origins = valid["origin_class"].tolist()

    # Build combined vector matrix: [old_vecs; new_vecs]
    old_vecs = np.array([aligned_old[old_w2i[w]] for w in words])
    new_vecs = np.array([model_new.wv[w] for w in words])
    combined = np.vstack([old_vecs, new_vecs])

    n_words = len(words)
    print(f"  Words: {n_words}")
    print(f"  Combined matrix: {combined.shape} (old + new stacked)")

    # Fit UMAP
    print(f"  Fitting UMAP (n_neighbors=15, min_dist=0.1, metric=cosine)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_SEED,
    )
    coords_2d = reducer.fit_transform(combined)

    # Split back
    old_coords = coords_2d[:n_words]
    new_coords = coords_2d[n_words:]

    print(f"  ✓ UMAP complete: {coords_2d.shape}")

    # Save coordinates
    records = []
    for i, word in enumerate(words):
        records.append({
            "word": word,
            "origin_class": origins[i],
            "x_old": round(float(old_coords[i, 0]), 6),
            "y_old": round(float(old_coords[i, 1]), 6),
            "x_new": round(float(new_coords[i, 0]), 6),
            "y_new": round(float(new_coords[i, 1]), 6),
            "drift_score": float(valid[valid["word"] == word]["drift_score"].iloc[0]),
        })

    umap_df = pd.DataFrame(records)
    umap_df.to_csv(UMAP_COORDS_CSV, index=False)

    print(f"  ✓ Saved to {UMAP_COORDS_CSV.name}")

    # Quick summary of spatial distribution
    for origin in ORIGIN_CLASSES:
        og = umap_df[umap_df["origin_class"] == origin]
        if len(og) > 0:
            # Mean displacement in 2D
            displacements = np.sqrt(
                (og["x_new"] - og["x_old"])**2 + (og["y_new"] - og["y_old"])**2
            )
            print(f"    {origin:<12} mean 2D displacement: {displacements.mean():.3f} "
                  f"(n={len(og)})")

    return umap_df


# ──────────────────────────────────────────────────────────────
#  Show drift rankings
# ──────────────────────────────────────────────────────────────
def show_drift():
    """Display current drift rankings."""
    if not DRIFT_SCORES_CSV.exists():
        print("✗ Run compute_drift.py first")
        return

    df = pd.read_csv(DRIFT_SCORES_CSV)
    valid = df[df["status"] == "OK"].sort_values("drift_score", ascending=False)

    print(f"\n{'=' * 60}")
    print(f"  Semantic Drift Rankings ({len(valid)} words)")
    print(f"{'=' * 60}")

    print(f"\n  {'#':<4} {'Word':<18} {'Origin':<12} {'Drift':>8} {'Sim':>8}")
    print(f"  {'─' * 52}")
    for i, (_, r) in enumerate(valid.iterrows(), 1):
        bar = "█" * int(r["drift_score"] * 25)
        print(f"  {i:<4} {r['word']:<18} {r['origin_class']:<12} "
              f"{r['drift_score']:>8.4f} {r['similarity']:>8.4f}  {bar}")


def show_word_neighbors(word: str):
    """Show one word's neighbors from the report."""
    if not NEIGHBOR_REPORT.exists():
        print("✗ Run compute_drift.py first")
        return

    text = NEIGHBOR_REPORT.read_text(encoding="utf-8")
    # Find the word section
    marker = f'"{word}"'
    idx = text.find(marker)
    if idx == -1:
        print(f"  '{word}' not found in neighbor analysis")
        return

    # Print from marker to next separator
    end = text.find("─" * 70, idx + 1)
    if end == -1:
        end = len(text)
    # Go back to the separator before the word
    start = text.rfind("─" * 70, 0, idx)
    if start == -1:
        start = idx

    print(text[start:end])


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Compute semantic drift scores and UMAP projections"
    )
    parser.add_argument("--show", action="store_true",
                        help="Show drift rankings")
    parser.add_argument("--neighbors", type=str, default=None,
                        help="Show neighbors for a specific word")
    args = parser.parse_args()

    if args.show:
        show_drift()
        return

    if args.neighbors:
        show_word_neighbors(args.neighbors)
        return

    print("=" * 60)
    print("  Phase 7: Semantic Drift Computation")
    print("=" * 60)

    t_start = time.time()

    # Load data
    model_old, model_new, aligned_old, old_words, old_w2i = load_aligned_data()

    # Step 7.1: Drift scores
    drift_df = compute_drift_scores(model_new, aligned_old, old_w2i)

    # Step 7.2: Neighbor analysis
    neighbor_analysis(
        model_old, model_new, aligned_old, old_words, old_w2i, drift_df
    )

    # Step 7.3: UMAP projection
    umap_df = compute_umap_projection(
        model_new, aligned_old, old_w2i, drift_df
    )

    elapsed = time.time() - t_start

    valid = drift_df[drift_df["status"] == "OK"]
    print(f"\n{'=' * 60}")
    print(f"  Phase 7 Complete!")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Words analyzed: {len(valid)}/100")
    print(f"  Mean drift: {valid['drift_score'].mean():.4f}")
    print(f"  Output:")
    print(f"    {DRIFT_SCORES_CSV.name}")
    print(f"    {NEIGHBOR_REPORT.name}")
    print(f"    {UMAP_COORDS_CSV.name}")
    print(f"  Next: python src/origin_analysis.py (Phase 8)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
