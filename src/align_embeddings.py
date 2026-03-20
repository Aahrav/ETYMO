"""
Embedding Alignment (Phase 6 — KEY STEP)
==========================================
Aligns the historical embedding space to the modern embedding space
using Orthogonal Procrustes, so that vectors from different time periods
become directly comparable.

Why this is critical:
  Word2Vec trains each model in its own coordinate system. Without
  alignment, cosine similarity between a 1800s vector and a 2000s vector
  is meaningless. Orthogonal Procrustes finds the optimal rotation matrix
  that maps the old space onto the new space, using "anchor" words
  (semantically stable words) as reference points.

  The rotation preserves internal distances (cosine similarities within
  each model stay the same), but makes cross-period similarities valid.

Steps:
  6.1  Extract shared vocabulary between both models
  6.2  Filter anchor words (must be in both vocabs)
  6.3  Build anchor matrices → Orthogonal Procrustes → rotation matrix W
  6.4  Apply W to ALL historical vectors → aligned vectors
  6.5  Validate alignment quality (anchor word drift should be near zero)

Usage:
    python src/align_embeddings.py             # Run full alignment
    python src/align_embeddings.py --validate  # Validate existing alignment
    python src/align_embeddings.py --compare WORD  # Compare word neighborhoods
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from gensim.models import Word2Vec

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    EMBEDDINGS_DIR, W2V_MODEL_OLD, W2V_MODEL_NEW,
    W2V_ALIGNED_OLD, RESULTS_DIR,
    ANCHOR_WORDS, SELECTED_WORDS_CSV,
    W2V_VECTOR_SIZE, ORIGIN_CLASSES,
)

ALIGNMENT_REPORT = RESULTS_DIR / "alignment_report.txt"


# ──────────────────────────────────────────────────────────────
#  Step 6.1: Extract shared vocabulary
# ──────────────────────────────────────────────────────────────
def extract_shared_vocab(model_old: Word2Vec, model_new: Word2Vec) -> set:
    """Compute vocabulary intersection between both models."""
    print("\n[Step 6.1] Extracting shared vocabulary...")

    vocab_old = set(model_old.wv.key_to_index.keys())
    vocab_new = set(model_new.wv.key_to_index.keys())
    shared = vocab_old & vocab_new

    print(f"  Historical vocab: {len(vocab_old):,}")
    print(f"  Modern vocab:     {len(vocab_new):,}")
    print(f"  Shared words:     {len(shared):,}")
    print(f"  Overlap ratio:    {len(shared) / len(vocab_old | vocab_new):.3f}")

    return shared


# ──────────────────────────────────────────────────────────────
#  Step 6.2: Filter anchor words
# ──────────────────────────────────────────────────────────────
def filter_anchors(shared_vocab: set) -> list[str]:
    """
    Filter anchor words to only those present in BOTH models.
    
    Anchor words must be:
      - In both vocabularies (already guaranteed by shared_vocab)
      - Semantically stable (configured in config.py)
    """
    print("\n[Step 6.2] Filtering anchor words...")

    usable = [w for w in ANCHOR_WORDS if w in shared_vocab]
    missing = [w for w in ANCHOR_WORDS if w not in shared_vocab]

    print(f"  Configured anchors: {len(ANCHOR_WORDS)}")
    print(f"  Usable anchors:     {len(usable)}")
    if missing:
        print(f"  Missing ({len(missing)}): {', '.join(missing[:10])}"
              f"{'...' if len(missing) > 10 else ''}")

    if len(usable) < 50:
        print(f"\n  ⚠ WARNING: Only {len(usable)} anchor words available.")
        print(f"    Procrustes alignment works best with 100+ anchors.")
        print(f"    Consider adding more anchor words to config.py")

    return usable


# ──────────────────────────────────────────────────────────────
#  Step 6.3: Orthogonal Procrustes Alignment
# ──────────────────────────────────────────────────────────────
def align_procrustes(
    model_old: Word2Vec,
    model_new: Word2Vec,
    anchor_words: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align historical embedding space to modern space using
    Orthogonal Procrustes.
    
    The algorithm finds rotation matrix W that minimizes:
        ||A_old @ W - A_new||_F
    
    where A_old and A_new are matrices of anchor word vectors.
    
    W is orthogonal (W^T @ W = I), meaning it only rotates the space
    without scaling or distortion — preserving internal cosine similarities.
    
    Returns:
        W: rotation matrix (dim × dim)
        aligned_old_vectors: all historical vectors rotated into modern space
    """
    print("\n[Step 6.3] Running Orthogonal Procrustes alignment...")
    print(f"  Using {len(anchor_words)} anchor words")
    print(f"  Rotating: Historical → Modern embedding space")

    # Build anchor matrices: (n_anchors × dim)
    A_old = np.array([model_old.wv[w] for w in anchor_words])
    A_new = np.array([model_new.wv[w] for w in anchor_words])

    print(f"  Anchor matrix shape: {A_old.shape}")

    # ── Normalize anchor vectors before Procrustes ──
    # This improves alignment quality by giving equal weight to all anchors
    A_old_norm = A_old / np.linalg.norm(A_old, axis=1, keepdims=True)
    A_new_norm = A_new / np.linalg.norm(A_new, axis=1, keepdims=True)

    # ── Solve: find W such that A_old_norm @ W ≈ A_new_norm ──
    W, scale = orthogonal_procrustes(A_old_norm, A_new_norm)

    print(f"  Rotation matrix W shape: {W.shape}")
    print(f"  Procrustes scale factor: {scale:.6f}")

    # Verify W is orthogonal: W^T @ W should be identity
    WtW = W.T @ W
    orthogonality_error = np.max(np.abs(WtW - np.eye(W2V_VECTOR_SIZE)))
    print(f"  Orthogonality error: {orthogonality_error:.2e} "
          f"({'✓ GOOD' if orthogonality_error < 1e-10 else '⚠ CHECK'})")

    # ── Apply rotation to ALL historical vectors ──
    print(f"\n  Applying rotation to all {len(model_old.wv):,} historical vectors...")
    all_old_vectors = model_old.wv.vectors  # (vocab_size × dim)
    aligned_old = all_old_vectors @ W       # Rotate into modern space

    print(f"  Aligned vectors shape: {aligned_old.shape}")

    return W, aligned_old


# ──────────────────────────────────────────────────────────────
#  Step 6.4: Save aligned vectors
# ──────────────────────────────────────────────────────────────
def save_aligned(
    aligned_vectors: np.ndarray,
    model_old: Word2Vec,
    W: np.ndarray,
):
    """Save aligned historical vectors and rotation matrix."""
    print("\n[Step 6.4] Saving aligned vectors...")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Save aligned vectors as numpy array
    np.save(str(W2V_ALIGNED_OLD), aligned_vectors)
    size_mb = W2V_ALIGNED_OLD.stat().st_size / (1024 * 1024)
    print(f"  ✓ Aligned vectors: {W2V_ALIGNED_OLD.name} ({size_mb:.1f} MB)")

    # Save rotation matrix
    W_path = EMBEDDINGS_DIR / "procrustes_rotation.npy"
    np.save(str(W_path), W)
    print(f"  ✓ Rotation matrix: {W_path.name}")

    # Save word-to-index mapping (needed to look up aligned vectors)
    vocab_path = EMBEDDINGS_DIR / "w2v_old_vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for word in model_old.wv.key_to_index:
            f.write(f"{word}\n")
    print(f"  ✓ Vocabulary index: {vocab_path.name}")


# ──────────────────────────────────────────────────────────────
#  Step 6.5: Validate alignment quality
# ──────────────────────────────────────────────────────────────
def validate_alignment(
    aligned_old: np.ndarray,
    model_old: Word2Vec,
    model_new: Word2Vec,
    anchor_words: list[str],
    shared_vocab: set,
) -> dict:
    """
    Validate alignment quality by checking:
      1. Anchor words should have VERY LOW drift (they're assumed stable)
      2. Known shifted words should have HIGHER drift
      3. Cosine similarities should be reasonable (not all 1.0 or 0.0)
    """
    print("\n[Step 6.5] Validating alignment quality...")

    # Helper: get aligned old vector by word
    old_word_to_idx = {w: i for i, w in enumerate(model_old.wv.key_to_index)}

    def cosine_sim(v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def get_aligned_old_vec(word):
        if word in old_word_to_idx:
            return aligned_old[old_word_to_idx[word]]
        return None

    # ── 1. Anchor word drift (should be LOW) ──
    print("\n  ── Anchor word drift (should be ~0, these are 'stable' words) ──")
    anchor_drifts = []
    for w in anchor_words:
        if w in model_new.wv and w in old_word_to_idx:
            vec_old_aligned = get_aligned_old_vec(w)
            vec_new = model_new.wv[w]
            sim = cosine_sim(vec_old_aligned, vec_new)
            drift = 1.0 - sim
            anchor_drifts.append(drift)

    mean_anchor_drift = np.mean(anchor_drifts)
    std_anchor_drift = np.std(anchor_drifts)
    max_anchor_drift = np.max(anchor_drifts)
    min_anchor_drift = np.min(anchor_drifts)

    print(f"    Mean anchor drift:  {mean_anchor_drift:.4f}")
    print(f"    Std anchor drift:   {std_anchor_drift:.4f}")
    print(f"    Range:              [{min_anchor_drift:.4f}, {max_anchor_drift:.4f}]")

    if mean_anchor_drift < 0.2:
        print(f"    ✓ Anchor drift is low — alignment looks good!")
    elif mean_anchor_drift < 0.4:
        print(f"    ⚠ Anchor drift is moderate — alignment is usable")
    else:
        print(f"    ✗ Anchor drift is HIGH — alignment may be poor")

    # Show top 5 most/least drifted anchors
    anchor_drift_pairs = []
    for w in anchor_words:
        if w in model_new.wv and w in old_word_to_idx:
            vec_old_aligned = get_aligned_old_vec(w)
            vec_new = model_new.wv[w]
            sim = cosine_sim(vec_old_aligned, vec_new)
            anchor_drift_pairs.append((w, 1.0 - sim))

    anchor_drift_pairs.sort(key=lambda x: x[1])

    print(f"\n    Most stable anchors (lowest drift):")
    for w, d in anchor_drift_pairs[:5]:
        print(f"      {w:<15} drift = {d:.4f}")

    print(f"\n    Least stable anchors (highest drift):")
    for w, d in anchor_drift_pairs[-5:]:
        print(f"      {w:<15} drift = {d:.4f}")

    # ── 2. Selected words drift preview ──
    print(f"\n  ── Selected words drift preview ──")
    if SELECTED_WORDS_CSV.exists():
        selected = pd.read_csv(SELECTED_WORDS_CSV)
        word_drifts = []

        for _, row in selected.iterrows():
            w = row["word"]
            if w in model_new.wv and w in old_word_to_idx:
                vec_old_aligned = get_aligned_old_vec(w)
                vec_new = model_new.wv[w]
                sim = cosine_sim(vec_old_aligned, vec_new)
                word_drifts.append((w, row["origin_class"], 1.0 - sim, sim))

        # Sort by drift (highest first)
        word_drifts.sort(key=lambda x: x[2], reverse=True)

        print(f"\n    Top 15 most-drifted selected words:")
        print(f"    {'Word':<18} {'Origin':<12} {'Drift':>8} {'Similarity':>10}")
        print(f"    {'─' * 50}")
        for w, origin, drift, sim in word_drifts[:15]:
            bar = "█" * int(drift * 30)
            print(f"    {w:<18} {origin:<12} {drift:>8.4f} {sim:>10.4f}  {bar}")

        print(f"\n    Top 10 most-stable selected words:")
        print(f"    {'Word':<18} {'Origin':<12} {'Drift':>8} {'Similarity':>10}")
        print(f"    {'─' * 50}")
        for w, origin, drift, sim in word_drifts[-10:]:
            print(f"    {w:<18} {origin:<12} {drift:>8.4f} {sim:>10.4f}")

        # Per-origin average drift
        print(f"\n    Per-origin average drift:")
        for origin in ORIGIN_CLASSES:
            origin_drifts = [d for w, o, d, s in word_drifts if o == origin]
            if origin_drifts:
                print(f"      {origin:<12} mean={np.mean(origin_drifts):.4f} "
                      f"(n={len(origin_drifts)})")

    stats = {
        "mean_anchor_drift": round(float(mean_anchor_drift), 4),
        "std_anchor_drift": round(float(std_anchor_drift), 4),
        "max_anchor_drift": round(float(max_anchor_drift), 4),
        "num_anchors": len(anchor_drifts),
        "num_selected_in_both": len(word_drifts) if SELECTED_WORDS_CSV.exists() else 0,
    }

    return stats


# ──────────────────────────────────────────────────────────────
#  Write alignment report
# ──────────────────────────────────────────────────────────────
def write_report(
    shared_vocab_size: int,
    num_anchors: int,
    alignment_stats: dict,
):
    """Write comprehensive alignment report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("  Phase 6: Embedding Alignment Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  Method: Orthogonal Procrustes (scipy.linalg.orthogonal_procrustes)")
    lines.append("  Direction: Historical → Modern embedding space")
    lines.append("")
    lines.append("  \"We rotate the historical embedding space to match the modern")
    lines.append("   space using Orthogonal Procrustes, preserving internal distances")
    lines.append("   while making cross-period comparisons valid.\"")
    lines.append("")
    lines.append(f"{'─' * 70}")
    lines.append(f"  Shared vocabulary:        {shared_vocab_size:,} words")
    lines.append(f"  Anchor words used:        {num_anchors}")
    lines.append(f"  Mean anchor drift:        {alignment_stats['mean_anchor_drift']:.4f}")
    lines.append(f"  Std anchor drift:         {alignment_stats['std_anchor_drift']:.4f}")
    lines.append(f"  Max anchor drift:         {alignment_stats['max_anchor_drift']:.4f}")
    lines.append(f"  Selected words available: {alignment_stats['num_selected_in_both']}/100")
    lines.append("")

    quality = "GOOD" if alignment_stats["mean_anchor_drift"] < 0.2 else \
              "FAIR" if alignment_stats["mean_anchor_drift"] < 0.4 else "POOR"
    lines.append(f"  Alignment quality: {quality}")
    lines.append(f"{'─' * 70}")

    report_text = "\n".join(lines)
    ALIGNMENT_REPORT.write_text(report_text, encoding="utf-8")
    print(f"\n  ✓ Report saved to {ALIGNMENT_REPORT.name}")
    print(f"\n{report_text}")


# ──────────────────────────────────────────────────────────────
#  Compare a word post-alignment
# ──────────────────────────────────────────────────────────────
def compare_word_aligned(word: str):
    """Show a word's neighbors in old (aligned) vs new space."""
    if not W2V_MODEL_OLD.exists() or not W2V_MODEL_NEW.exists():
        print("✗ Both models must be trained first.")
        return
    if not W2V_ALIGNED_OLD.exists():
        print("✗ Alignment must be run first.")
        return

    model_old = Word2Vec.load(str(W2V_MODEL_OLD))
    model_new = Word2Vec.load(str(W2V_MODEL_NEW))
    aligned_old = np.load(str(W2V_ALIGNED_OLD))
    old_words = list(model_old.wv.key_to_index.keys())
    old_word_to_idx = {w: i for i, w in enumerate(old_words)}

    def cosine_sim(v1, v2):
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    print(f"\n{'=' * 60}")
    print(f"  '{word}' — Aligned Cross-Period Comparison")
    print(f"{'=' * 60}")

    # Historical neighbors (using aligned vectors)
    if word in old_word_to_idx:
        target_vec = aligned_old[old_word_to_idx[word]]
        # Compute cosine similarity with all aligned old vectors
        norms = np.linalg.norm(aligned_old, axis=1)
        norms[norms == 0] = 1
        sims = aligned_old @ target_vec / (norms * np.linalg.norm(target_vec))
        top_idxs = np.argsort(sims)[::-1][1:11]  # Skip self

        print(f"\n  Historical (1800–1900) — aligned space:")
        for i, idx in enumerate(top_idxs, 1):
            print(f"    {i:>2}. {old_words[idx]:<20} {sims[idx]:.4f}")
    else:
        print(f"\n  '{word}' not in historical vocabulary")

    # Modern neighbors (native space)
    if word in model_new.wv:
        print(f"\n  Modern (2000–2020):")
        for i, (w, s) in enumerate(model_new.wv.most_similar(word, topn=10), 1):
            print(f"    {i:>2}. {w:<20} {s:.4f}")
    else:
        print(f"\n  '{word}' not in modern vocabulary")

    # Cross-period similarity
    if word in old_word_to_idx and word in model_new.wv:
        v_old = aligned_old[old_word_to_idx[word]]
        v_new = model_new.wv[word]
        sim = cosine_sim(v_old, v_new)
        drift = 1 - sim
        print(f"\n  Cross-period cosine similarity: {sim:.4f}")
        print(f"  Semantic drift score:           {drift:.4f}")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Align embedding spaces using Orthogonal Procrustes"
    )
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing alignment")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare a word across aligned spaces")
    args = parser.parse_args()

    if args.compare:
        compare_word_aligned(args.compare)
        return

    print("=" * 60)
    print("  Phase 6: Embedding Alignment (KEY STEP)")
    print("  Orthogonal Procrustes: rotate old → new space")
    print("=" * 60)

    t_start = time.time()

    # ── Load models ──
    print("\n  Loading models...")
    if not W2V_MODEL_OLD.exists() or not W2V_MODEL_NEW.exists():
        print("  ✗ Both models must be trained first (Phase 5)")
        sys.exit(1)

    model_old = Word2Vec.load(str(W2V_MODEL_OLD))
    model_new = Word2Vec.load(str(W2V_MODEL_NEW))
    print(f"  ✓ Historical: {len(model_old.wv):,} words")
    print(f"  ✓ Modern:     {len(model_new.wv):,} words")

    # ── Step 6.1: Shared vocabulary ──
    shared = extract_shared_vocab(model_old, model_new)

    # ── Step 6.2: Filter anchors ──
    anchors = filter_anchors(shared)

    # ── Step 6.3: Procrustes alignment ──
    W, aligned_old = align_procrustes(model_old, model_new, anchors)

    # ── Step 6.4: Save ──
    save_aligned(aligned_old, model_old, W)

    # ── Step 6.5: Validate ──
    stats = validate_alignment(
        aligned_old, model_old, model_new, anchors, shared
    )

    # ── Report ──
    write_report(len(shared), len(anchors), stats)

    elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"  Phase 6 Complete!")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Anchors used:   {len(anchors)}")
    print(f"  Mean anchor drift: {stats['mean_anchor_drift']:.4f}")
    print(f"  Selected words:    {stats['num_selected_in_both']}/100")
    print(f"  Next: python src/compute_drift.py (Phase 7)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
