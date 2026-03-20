"""
Train Word Embeddings (Phase 5)
=================================
Trains two Word2Vec Skip-gram models — one per time period — on the
preprocessed corpora from Phase 4. Then runs comprehensive validation
to ensure embedding quality before alignment.

Hyperparameter choices (justified in implementation_plan.md):
  sg=1       Skip-gram: better for small corpora, captures rare words
  dim=100    Sweet spot for 5–10M token corpora (300d would be under-trained)
  window=5   Balances syntactic + semantic context
  min_count=10  Filters noisy rare words (yields ~20K–50K vocab)
  epochs=15  Compensates for smaller corpus (default is 5)
  negative=10  Higher negative sampling improves quality on small data

Usage:
    python src/train_embeddings.py               # Train both models
    python src/train_embeddings.py --old          # Train historical only
    python src/train_embeddings.py --new          # Train modern only
    python src/train_embeddings.py --validate     # Validate existing models
    python src/train_embeddings.py --compare WORD # Compare a word across periods
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
    RANDOM_SEED,
    CORPUS_OLD, CORPUS_NEW,
    EMBEDDINGS_DIR, W2V_MODEL_OLD, W2V_MODEL_NEW,
    EMBEDDING_REPORT, RESULTS_DIR,
    W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT,
    W2V_SG, W2V_EPOCHS, W2V_NEGATIVE, W2V_WORKERS,
    SELECTED_WORDS_CSV, ANCHOR_WORDS,
    ORIGIN_CLASSES,
)


# ──────────────────────────────────────────────────────────────
#  Corpus iterator (memory-efficient line-by-line reading)
# ──────────────────────────────────────────────────────────────
class CorpusIterator:
    """
    Iterate over a preprocessed corpus file, yielding one sentence
    (as a list of tokens) per iteration.
    
    Memory-efficient: reads one line at a time, never loads the
    full file into memory.
    """
    def __init__(self, corpus_path: Path):
        self.corpus_path = corpus_path
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    def __iter__(self):
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens

    def count_stats(self) -> dict:
        """Count sentences, tokens, and unique words."""
        sentences = 0
        tokens = 0
        vocab = set()
        for sent in self:
            sentences += 1
            tokens += len(sent)
            vocab.update(sent)
        return {
            "sentences": sentences,
            "tokens": tokens,
            "unique_words": len(vocab),
        }


# ──────────────────────────────────────────────────────────────
#  Step 5.1: Train a Word2Vec model
# ──────────────────────────────────────────────────────────────
def train_model(
    corpus_path: Path,
    model_path: Path,
    label: str,
) -> Word2Vec:
    """
    Train a Word2Vec Skip-gram model on the given corpus.
    
    Uses the hyperparameters defined in config.py — these are
    identical for both models to ensure fair comparison.
    """
    print(f"\n[Step 5.1] Training Word2Vec — {label}")
    print(f"  Corpus: {corpus_path.name}")
    print(f"  Output: {model_path.name}")
    print(f"\n  Hyperparameters:")
    print(f"    algorithm:      Skip-gram (sg={W2V_SG})")
    print(f"    vector_size:    {W2V_VECTOR_SIZE}")
    print(f"    window:         {W2V_WINDOW}")
    print(f"    min_count:      {W2V_MIN_COUNT}")
    print(f"    epochs:         {W2V_EPOCHS}")
    print(f"    negative:       {W2V_NEGATIVE}")
    print(f"    workers:        {W2V_WORKERS}")
    print(f"    seed:           {RANDOM_SEED}")

    # Check if model already exists
    if model_path.exists():
        print(f"\n  ℹ Model already exists at {model_path.name}")
        print(f"    Loading existing model...")
        model = Word2Vec.load(str(model_path))
        print(f"    Vocabulary: {len(model.wv):,} words")
        return model

    # Create corpus iterator
    corpus = CorpusIterator(corpus_path)

    # Quick corpus stats
    print(f"\n  Scanning corpus...")
    stats = corpus.count_stats()
    print(f"    Sentences: {stats['sentences']:,}")
    print(f"    Tokens:    {stats['tokens']:,}")
    print(f"    Unique:    {stats['unique_words']:,}")

    # Train
    print(f"\n  Training (this may take 2–5 minutes)...")
    t_start = time.time()

    model = Word2Vec(
        sentences=CorpusIterator(corpus_path),  # Fresh iterator
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        sg=W2V_SG,
        epochs=W2V_EPOCHS,
        negative=W2V_NEGATIVE,
        workers=W2V_WORKERS,
        seed=RANDOM_SEED,
        compute_loss=True,
    )

    elapsed = time.time() - t_start

    print(f"\n  ✓ Training complete in {elapsed:.1f}s")
    print(f"    Vocabulary: {len(model.wv):,} words")
    print(f"    Training loss: {model.get_latest_training_loss():.0f}")

    # Save model
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {model_path.name} ({size_mb:.1f} MB)")

    return model


# ──────────────────────────────────────────────────────────────
#  Step 5.2: Validate embeddings
# ──────────────────────────────────────────────────────────────
def validate_model(model: Word2Vec, label: str) -> dict:
    """
    Run sanity checks on a trained Word2Vec model:
      1. Vocabulary size is reasonable (expect 15K–60K for 5–10M tokens)
      2. most_similar() returns semantically sensible results
      3. Vector norms are reasonable (not degenerate)
    """
    print(f"\n  ── Validating: {label} ──")
    wv = model.wv
    stats = {"label": label, "vocab_size": len(wv)}

    print(f"    Vocabulary: {len(wv):,} words")

    # Sanity check: most_similar for well-known words
    test_words = ["king", "water", "good", "time", "work"]
    print(f"\n    Semantic sanity checks:")
    for word in test_words:
        if word in wv:
            similar = wv.most_similar(word, topn=5)
            neighbors = ", ".join(f"{w} ({s:.3f})" for w, s in similar)
            print(f"      {word:>8} → {neighbors}")
        else:
            print(f"      {word:>8} → NOT IN VOCAB")

    # Analogy test (classic: king - man + woman ≈ queen)
    try:
        analogy = wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
        analogy_results = ", ".join(f"{w} ({s:.3f})" for w, s in analogy)
        print(f"\n    Analogy test: king - man + woman = {analogy_results}")
        stats["analogy_top"] = analogy[0][0]
    except KeyError:
        print(f"\n    Analogy test: skipped (missing words)")
        stats["analogy_top"] = "N/A"

    # Vector statistics
    all_vectors = wv.vectors
    norms = np.linalg.norm(all_vectors, axis=1)
    stats["mean_norm"] = round(float(np.mean(norms)), 4)
    stats["std_norm"] = round(float(np.std(norms)), 4)
    stats["min_norm"] = round(float(np.min(norms)), 4)
    stats["max_norm"] = round(float(np.max(norms)), 4)

    print(f"\n    Vector norms: mean={stats['mean_norm']:.3f}, "
          f"std={stats['std_norm']:.3f}, "
          f"range=[{stats['min_norm']:.3f}, {stats['max_norm']:.3f}]")

    return stats


def verify_selected_words(model_old: Word2Vec, model_new: Word2Vec) -> dict:
    """
    Verify that all 100 selected words appear in BOTH models' vocabularies.
    Words missing from either model can't be used for drift analysis.
    """
    print(f"\n[Step 5.2] Verifying selected words in both models...")

    if not SELECTED_WORDS_CSV.exists():
        print(f"  ⚠ selected_words.csv not found")
        return {}

    selected = pd.read_csv(SELECTED_WORDS_CSV)
    words = selected["word"].tolist()

    in_old = [w for w in words if w in model_old.wv]
    in_new = [w for w in words if w in model_new.wv]
    in_both = [w for w in words if w in model_old.wv and w in model_new.wv]

    missing_old = [w for w in words if w not in model_old.wv]
    missing_new = [w for w in words if w not in model_new.wv]
    missing_any = sorted(set(missing_old + missing_new))

    print(f"\n  Selected words coverage:")
    print(f"    In historical model: {len(in_old)}/{len(words)}")
    print(f"    In modern model:     {len(in_new)}/{len(words)}")
    print(f"    In BOTH models:      {len(in_both)}/{len(words)}")

    if missing_old:
        print(f"\n    ⚠ Missing from historical: {', '.join(missing_old)}")
    if missing_new:
        print(f"\n    ⚠ Missing from modern: {', '.join(missing_new)}")

    if len(in_both) == len(words):
        print(f"\n  ✓ All {len(words)} selected words are in both vocabularies!")
    elif len(in_both) >= 80:
        print(f"\n  ⚠ {len(missing_any)} words missing, but {len(in_both)} available — sufficient for analysis")
    else:
        print(f"\n  ✗ Only {len(in_both)} words in both — consider revising word selection")

    # Per-origin breakdown
    print(f"\n  Per-origin coverage (in BOTH models):")
    for origin in ORIGIN_CLASSES:
        origin_words = selected[selected["origin_class"] == origin]["word"].tolist()
        origin_both = [w for w in origin_words if w in model_old.wv and w in model_new.wv]
        print(f"    {origin:<12} {len(origin_both)}/{len(origin_words)}")

    return {
        "total_selected": len(words),
        "in_historical": len(in_old),
        "in_modern": len(in_new),
        "in_both": len(in_both),
        "missing_old": missing_old,
        "missing_new": missing_new,
    }


def verify_anchor_words(model_old: Word2Vec, model_new: Word2Vec) -> dict:
    """Verify anchor words for Phase 6 alignment."""
    print(f"\n  Verifying anchor words for alignment...")

    in_both = [w for w in ANCHOR_WORDS if w in model_old.wv and w in model_new.wv]
    missing = [w for w in ANCHOR_WORDS if w not in model_old.wv or w not in model_new.wv]

    print(f"    Configured anchors: {len(ANCHOR_WORDS)}")
    print(f"    In BOTH models:     {len(in_both)}")
    if missing:
        print(f"    Missing: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")
    print(f"    ✓ {len(in_both)} anchor words ready for Procrustes alignment")

    return {"configured": len(ANCHOR_WORDS), "usable": len(in_both), "missing": missing}


def compute_vocab_overlap(model_old: Word2Vec, model_new: Word2Vec) -> dict:
    """Compute vocabulary overlap between the two models."""
    print(f"\n  Computing vocabulary overlap...")

    vocab_old = set(model_old.wv.key_to_index.keys())
    vocab_new = set(model_new.wv.key_to_index.keys())
    shared = vocab_old & vocab_new
    only_old = vocab_old - vocab_new
    only_new = vocab_new - vocab_old

    print(f"    Historical vocab: {len(vocab_old):,}")
    print(f"    Modern vocab:     {len(vocab_new):,}")
    print(f"    Shared words:     {len(shared):,}")
    print(f"    Only historical:  {len(only_old):,}")
    print(f"    Only modern:      {len(only_new):,}")
    print(f"    Overlap ratio:    {len(shared) / len(vocab_old | vocab_new):.3f}")

    return {
        "vocab_old": len(vocab_old),
        "vocab_new": len(vocab_new),
        "shared": len(shared),
        "only_old": len(only_old),
        "only_new": len(only_new),
        "overlap_ratio": round(len(shared) / len(vocab_old | vocab_new), 4),
    }


# ──────────────────────────────────────────────────────────────
#  Write report
# ──────────────────────────────────────────────────────────────
def write_report(
    stats_old: dict,
    stats_new: dict,
    word_coverage: dict,
    anchor_info: dict,
    overlap: dict,
):
    """Write comprehensive embedding quality report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("  Phase 5: Word Embedding Report")
    lines.append("=" * 70)
    lines.append("")

    # Hyperparameters
    lines.append("  Hyperparameters (identical for both models):")
    lines.append(f"    Algorithm:    Skip-gram (sg={W2V_SG})")
    lines.append(f"    Dimensions:   {W2V_VECTOR_SIZE}")
    lines.append(f"    Window:       {W2V_WINDOW}")
    lines.append(f"    Min count:    {W2V_MIN_COUNT}")
    lines.append(f"    Epochs:       {W2V_EPOCHS}")
    lines.append(f"    Neg samples:  {W2V_NEGATIVE}")
    lines.append("")

    # Model stats
    lines.append(f"{'─' * 70}")
    lines.append(f"  {'Metric':<35} {'Historical':>15} {'Modern':>15}")
    lines.append(f"{'─' * 70}")
    lines.append(f"  {'Vocabulary size':<35} {stats_old['vocab_size']:>15,} {stats_new['vocab_size']:>15,}")
    lines.append(f"  {'Mean vector norm':<35} {stats_old['mean_norm']:>15.3f} {stats_new['mean_norm']:>15.3f}")
    lines.append(f"  {'Std vector norm':<35} {stats_old['std_norm']:>15.3f} {stats_new['std_norm']:>15.3f}")
    lines.append(f"  {'Analogy (king-man+woman)':<35} {stats_old.get('analogy_top', 'N/A'):>15} {stats_new.get('analogy_top', 'N/A'):>15}")
    lines.append("")

    # Vocabulary overlap
    lines.append(f"{'─' * 70}")
    lines.append(f"  Vocabulary Overlap")
    lines.append(f"{'─' * 70}")
    lines.append(f"  Shared words:    {overlap['shared']:>7,}")
    lines.append(f"  Only historical: {overlap['only_old']:>7,}")
    lines.append(f"  Only modern:     {overlap['only_new']:>7,}")
    lines.append(f"  Overlap ratio:   {overlap['overlap_ratio']:.4f}")
    lines.append("")

    # Selected word coverage
    lines.append(f"{'─' * 70}")
    lines.append(f"  Selected Word Coverage")
    lines.append(f"{'─' * 70}")
    lines.append(f"  Words in both models: {word_coverage.get('in_both', 'N/A')}/{word_coverage.get('total_selected', 'N/A')}")
    missing_all = set(word_coverage.get('missing_old', [])) | set(word_coverage.get('missing_new', []))
    if missing_all:
        lines.append(f"  Missing words: {', '.join(sorted(missing_all))}")
    lines.append("")

    # Anchor word readiness
    lines.append(f"{'─' * 70}")
    lines.append(f"  Anchor Words (Phase 6 Alignment)")
    lines.append(f"{'─' * 70}")
    lines.append(f"  Configured: {anchor_info['configured']}")
    lines.append(f"  Usable:     {anchor_info['usable']}")
    if anchor_info['missing']:
        lines.append(f"  Missing:    {', '.join(anchor_info['missing'][:15])}")

    report_text = "\n".join(lines)

    EMBEDDING_REPORT.write_text(report_text, encoding="utf-8")
    print(f"\n  ✓ Report saved to {EMBEDDING_REPORT.name}")

    print(f"\n{report_text}")


# ──────────────────────────────────────────────────────────────
#  Compare a word across periods
# ──────────────────────────────────────────────────────────────
def compare_word(word: str):
    """Show a word's neighborhood in both models side by side."""
    if not W2V_MODEL_OLD.exists() or not W2V_MODEL_NEW.exists():
        print("✗ Both models must be trained first.")
        return

    model_old = Word2Vec.load(str(W2V_MODEL_OLD))
    model_new = Word2Vec.load(str(W2V_MODEL_NEW))

    print(f"\n{'=' * 55}")
    print(f"  Comparing '{word}' across time periods")
    print(f"{'=' * 55}")

    for label, model in [("Historical (1800–1900)", model_old),
                         ("Modern (2000–2020)", model_new)]:
        print(f"\n  ── {label} ──")
        if word not in model.wv:
            print(f"    '{word}' not in vocabulary")
            continue
        similar = model.wv.most_similar(word, topn=10)
        for i, (w, s) in enumerate(similar, 1):
            bar = "█" * int(s * 30)
            print(f"    {i:>2}. {w:<20} {s:.4f} {bar}")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Train Word2Vec embeddings for both time periods"
    )
    parser.add_argument("--old", action="store_true",
                        help="Train historical model only")
    parser.add_argument("--new", action="store_true",
                        help="Train modern model only")
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing models")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare a word across periods")
    args = parser.parse_args()

    if args.compare:
        compare_word(args.compare)
        return

    print("=" * 60)
    print("  Phase 5: Train Word Embeddings")
    print("  Goal: Word2Vec Skip-gram for both time periods")
    print("=" * 60)

    t_start = time.time()

    train_old = args.old or args.validate or (not args.old and not args.new)
    train_new = args.new or args.validate or (not args.old and not args.new)

    # ── Train / Load models ──
    model_old = None
    model_new = None

    if train_old:
        if args.validate and W2V_MODEL_OLD.exists():
            print(f"\n  Loading existing historical model...")
            model_old = Word2Vec.load(str(W2V_MODEL_OLD))
        else:
            model_old = train_model(CORPUS_OLD, W2V_MODEL_OLD, "Historical (1800–1900)")

    if train_new:
        if args.validate and W2V_MODEL_NEW.exists():
            print(f"\n  Loading existing modern model...")
            model_new = Word2Vec.load(str(W2V_MODEL_NEW))
        else:
            model_new = train_model(CORPUS_NEW, W2V_MODEL_NEW, "Modern (2000–2020)")

    # ── Validate ──
    stats_old = None
    stats_new = None

    if model_old:
        stats_old = validate_model(model_old, "Historical (1800–1900)")

    if model_new:
        stats_new = validate_model(model_new, "Modern (2000–2020)")

    # ── Cross-model validation ──
    word_coverage = {}
    anchor_info = {}
    overlap = {}

    if model_old and model_new:
        word_coverage = verify_selected_words(model_old, model_new)
        anchor_info = verify_anchor_words(model_old, model_new)
        overlap = compute_vocab_overlap(model_old, model_new)

        # Write comprehensive report
        write_report(stats_old, stats_new, word_coverage, anchor_info, overlap)

    elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"  Phase 5 Complete!")
    print(f"  Time: {elapsed:.1f}s")
    if model_old:
        print(f"  Historical: {len(model_old.wv):,} words → {W2V_MODEL_OLD.name}")
    if model_new:
        print(f"  Modern:     {len(model_new.wv):,} words → {W2V_MODEL_NEW.name}")
    if word_coverage:
        print(f"  Selected words in both: {word_coverage.get('in_both', '?')}/100")
    if anchor_info:
        print(f"  Anchor words ready:     {anchor_info.get('usable', '?')}/{anchor_info.get('configured', '?')}")
    print(f"  Next: python src/align_embeddings.py (Phase 6)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
