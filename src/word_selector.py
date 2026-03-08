"""
Word Selector (Phase 3)
========================
Selects ~100 high-confidence words (25 per etymological origin)
for semantic drift tracking across time periods.

Strategy (3 tiers of word sources):
  Tier 1: Curated seed words with KNOWN semantic shift histories
  Tier 2: High-confidence classifier predictions from NLTK words corpus
  Tier 3: Words already in our training dataset (verified origins)

Each tier fills remaining slots per origin class, ensuring a balanced
final selection of ~25 words per class.

Usage:
    python src/word_selector.py                    # Run full selection
    python src/word_selector.py --show             # Display current selection
    python src/word_selector.py --stats            # Show detailed statistics
"""

import sys
import argparse
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    ORIGIN_CLASSES, RANDOM_SEED,
    WORDS_PER_ORIGIN, TOTAL_SHIFT_WORDS,
    CLASSIFIER_CONFIDENCE_THRESHOLD,
    CLASSIFIER_MODEL,
    ORIGIN_DATASET, TRAIN_DATASET, TEST_DATASET,
    SELECTED_WORDS_CSV, SELECTION_REPORT,
    MIN_WORD_LENGTH, MAX_WORD_LENGTH,
    SEED_WORDS,
    DATA_DIR, RESULTS_DIR,
)

# Import the anchor function from classifier module
from src.classifier import anchor


# ──────────────────────────────────────────────────────────────
#  Step 1: Load the trained classifier pipeline
# ──────────────────────────────────────────────────────────────
def load_classifier():
    """Load the saved classifier pipeline (vectorizer + model)."""
    print("\n[Step 1] Loading trained classifier pipeline...")

    if not CLASSIFIER_MODEL.exists():
        print(f"  ✗ Model not found at {CLASSIFIER_MODEL}")
        print(f"    Run `python src/classifier.py` first (Phase 2).")
        sys.exit(1)

    pipeline = joblib.load(CLASSIFIER_MODEL)
    classes = list(pipeline.classes_)
    print(f"  ✓ Loaded pipeline from {CLASSIFIER_MODEL.name}")
    print(f"    Classes: {classes}")

    return pipeline, classes


# ──────────────────────────────────────────────────────────────
#  Step 2: Load candidate word lists
# ──────────────────────────────────────────────────────────────
def load_candidate_words() -> tuple[set[str], dict[str, int], set[str]]:
    """
    Load candidate words from NLTK words corpus, build a word-frequency
    dictionary from the Brown corpus, and load stopwords.
    
    Returns:
        candidates: set of English words (lowercase, alpha, filtered)
        word_freq: dict mapping word -> frequency count in Brown corpus
        stopwords: set of English stop words to exclude from Tier 2
    """
    print("\n[Step 2] Loading candidate words, frequency data, and stopwords...")

    import nltk

    # ── Load word list ──
    try:
        nltk.download("words", quiet=True)
        from nltk.corpus import words as nltk_words
        raw = nltk_words.words()
    except Exception as e:
        print(f"  ⚠ NLTK words corpus unavailable: {e}")
        return set(), {}, set()

    candidates = set()
    for w in raw:
        w_lower = w.lower()
        if (
            w_lower.isalpha()
            and MIN_WORD_LENGTH <= len(w_lower) <= MAX_WORD_LENGTH
            and w[0].islower()  # skip proper nouns
        ):
            candidates.add(w_lower)

    print(f"  ✓ {len(candidates):,} unique candidate words after filtering")

    # ── Build word frequency from Brown corpus ──
    # The Brown corpus is ~1M tokens of real English text across 15 genres.
    # Words that appear here are ACTUALLY USED in English — this filters
    # out dictionary curiosities like 'weam', 'koku', 'fow' etc.
    try:
        nltk.download("brown", quiet=True)
        from nltk.corpus import brown
        word_freq = Counter(w.lower() for w in brown.words() if w.isalpha())
        print(f"  ✓ Brown corpus frequency: {len(word_freq):,} unique words")
        print(f"    (words NOT in Brown corpus will be excluded from Tier 2)")
    except Exception as e:
        print(f"  ⚠ Brown corpus unavailable: {e}")
        word_freq = {}

    # ── Load stopwords ──
    # Function words (the, is, when, his, from...) are the most frequent
    # words in any corpus, but they carry no semantic content and don't
    # undergo meaningful semantic drift. We must exclude them.
    try:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as nltk_stopwords
        sw = set(nltk_stopwords.words("english"))
        # Add extra function words / closed-class words that NLTK misses
        extra_stop = {
            "would", "could", "should", "might", "must", "shall",
            "also", "yet", "still", "however", "although", "though",
            "perhaps", "quite", "rather", "already", "thus", "hence",
            "therefore", "moreover", "indeed", "nevertheless",
            "upon", "whose", "whom", "thereof", "whereby", "wherein",
            "somewhat", "sometimes", "meanwhile", "everything",
            "anything", "nothing", "something", "someone",
            "everyone", "anyone", "nobody", "somebody",
            "first", "second", "third", "last", "next",
        }
        sw.update(extra_stop)
        print(f"  ✓ Stopwords: {len(sw)} function words to exclude")
    except Exception as e:
        print(f"  ⚠ Stopwords unavailable: {e}")
        sw = set()

    return candidates, word_freq, sw


def load_dataset_words() -> pd.DataFrame:
    """Load the full origin dataset (all labeled words from EtymWN)."""
    print("\n  Loading labeled dataset words...")

    if not ORIGIN_DATASET.exists():
        print(f"  ✗ Dataset not found at {ORIGIN_DATASET}")
        sys.exit(1)

    df = pd.read_csv(ORIGIN_DATASET)
    print(f"  ✓ {len(df):,} labeled words from {ORIGIN_DATASET.name}")
    return df


# ──────────────────────────────────────────────────────────────
#  Step 3: Classify candidates and filter by confidence
# ──────────────────────────────────────────────────────────────
def classify_candidates(
    pipeline,
    classes: list[str],
    candidates: set[str],
    known_words: set[str],
    word_freq: dict[str, int],
    stopwords: set[str],
) -> pd.DataFrame:
    """
    Run the classifier on NLTK candidates that are:
      - NOT already in our labeled dataset
      - PRESENT in the Brown corpus (i.e., actually used in real English)
      - NOT stopwords or function words
      - At least 4 characters long (content words, not 'the', 'is', etc.)
    
    Return a DataFrame of high-confidence predictions with frequency data.
    """
    print("\n[Step 3] Classifying candidate words...")

    # Remove words already in our training data
    new_candidates = sorted(candidates - known_words)
    print(f"  Candidates not in dataset: {len(new_candidates):,}")

    # Filter to only words that appear in Brown corpus (real English usage)
    if word_freq:
        freq_filtered = [w for w in new_candidates if w in word_freq]
        print(f"  After Brown corpus frequency filter: {len(freq_filtered):,} "
              f"(removed {len(new_candidates) - len(freq_filtered):,} obscure words)")
        new_candidates = freq_filtered

    # Remove stopwords and require min 4 chars for content words
    if stopwords:
        content_filtered = [
            w for w in new_candidates
            if w not in stopwords and len(w) >= 4
        ]
        print(f"  After stopword + length filter: {len(content_filtered):,} "
              f"(removed {len(new_candidates) - len(content_filtered):,} function/short words)")
        new_candidates = content_filtered

    if not new_candidates:
        return pd.DataFrame(columns=["word", "origin_class", "confidence", "source", "frequency"])

    # Classify in batches
    batch_size = 5000
    all_results = []

    for i in range(0, len(new_candidates), batch_size):
        batch = new_candidates[i : i + batch_size]
        anchored = [anchor(w) for w in batch]

        probas = pipeline.predict_proba(anchored)
        preds = pipeline.predict(anchored)

        for word, pred, proba in zip(batch, preds, probas):
            pred_idx = classes.index(pred)
            conf = proba[pred_idx]

            if conf >= CLASSIFIER_CONFIDENCE_THRESHOLD:
                all_results.append({
                    "word": word,
                    "origin_class": pred,
                    "confidence": round(float(conf), 4),
                    "source": "classifier",
                    "frequency": word_freq.get(word, 0),
                })

        if (i // batch_size) % 10 == 0:
            print(f"    Processed {min(i + batch_size, len(new_candidates)):,}/{len(new_candidates):,}...")

    df = pd.DataFrame(all_results)
    print(f"\n  ✓ {len(df):,} words with confidence ≥ {CLASSIFIER_CONFIDENCE_THRESHOLD}")

    if len(df) > 0:
        for cls in ORIGIN_CLASSES:
            count = len(df[df["origin_class"] == cls])
            print(f"    {cls:<12} {count:>6} high-confidence words")

    return df


# ──────────────────────────────────────────────────────────────
#  Step 4: Build the final selection (3-tier approach)
# ──────────────────────────────────────────────────────────────
def select_words(
    pipeline,
    classes: list[str],
    classifier_candidates: pd.DataFrame,
    dataset_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select ~25 words per origin class using a 3-tier priority system:
    
    Tier 1 (highest priority): Curated seed words with known semantic shifts.
           These are the MOST interesting words for the analysis —
           "mouse", "virus", "phone", "algebra" etc.
           
    Tier 2: High-confidence classifier predictions from NLTK.
           Common English words the model is very sure about, sorted
           by confidence. Prefer shorter, more common words.
           
    Tier 3: Words from our labeled training dataset (EtymWN-verified).
           These have TRUE labels, not predicted. Last resort to fill
           remaining slots.
    """
    print("\n[Step 4] Building final word selection (3-tier approach)...")

    selected = []  # List of dicts: {word, origin_class, confidence, source, tier}
    used_words = set()

    for origin in ORIGIN_CLASSES:
        print(f"\n  ── {origin} ──")
        slots_remaining = WORDS_PER_ORIGIN

        # ── TIER 1: Curated seed words ──
        seeds = SEED_WORDS.get(origin, [])
        tier1_count = 0

        for word in seeds:
            if slots_remaining <= 0:
                break
            if word in used_words:
                continue

            # Classify the seed word to get confidence
            anchored = anchor(word)
            proba = pipeline.predict_proba([anchored])[0]
            pred = pipeline.predict([anchored])[0]
            pred_idx = classes.index(pred)
            conf = proba[pred_idx]

            # Use the SEED's intended origin, not the classifier prediction
            # (the whole point of seeds is they have known origins)
            selected.append({
                "word": word,
                "origin_class": origin,
                "confidence": round(float(conf), 4),
                "source": "curated_seed",
                "tier": 1,
            })
            used_words.add(word)
            slots_remaining -= 1
            tier1_count += 1

        print(f"    Tier 1 (seeds):      {tier1_count} words")

        # ── TIER 2: High-confidence classifier predictions ──
        tier2_count = 0
        if slots_remaining > 0 and len(classifier_candidates) > 0:
            origin_candidates = classifier_candidates[
                (classifier_candidates["origin_class"] == origin)
                & (~classifier_candidates["word"].isin(used_words))
            ].copy()

            if len(origin_candidates) > 0:
                # Scoring: heavily weight FREQUENCY (common words first),
                # then confidence, with a small penalty for very long words.
                # log1p(freq) ensures common words like "wind" (freq=200+)
                # score far above rare words like "wod" (freq=1).
                import math
                origin_candidates["freq"] = origin_candidates["frequency"].apply(
                    lambda f: math.log1p(f)
                )
                origin_candidates["score"] = (
                    origin_candidates["freq"] * 0.6          # frequency dominates
                    + origin_candidates["confidence"] * 0.3  # confidence matters
                    - origin_candidates["word"].str.len() * 0.01  # slight length penalty
                )
                origin_candidates = origin_candidates.sort_values(
                    "score", ascending=False
                )

                for _, row in origin_candidates.head(slots_remaining).iterrows():
                    selected.append({
                        "word": row["word"],
                        "origin_class": origin,
                        "confidence": row["confidence"],
                        "source": "classifier_nltk",
                        "tier": 2,
                    })
                    used_words.add(row["word"])
                    slots_remaining -= 1
                    tier2_count += 1

        print(f"    Tier 2 (classifier): {tier2_count} words")

        # ── TIER 3: Labeled dataset words (EtymWN verified) ──
        tier3_count = 0
        if slots_remaining > 0:
            dataset_origin = dataset_df[
                (dataset_df["origin_class"] == origin)
                & (~dataset_df["word"].isin(used_words))
                & (dataset_df["word"].str.len() >= MIN_WORD_LENGTH)
                & (dataset_df["word"].str.len() <= MAX_WORD_LENGTH)
            ].copy()

            if len(dataset_origin) > 0:
                # Prefer common-looking words (shorter, no unusual characters)
                dataset_origin["score"] = -dataset_origin["word"].str.len()
                dataset_origin = dataset_origin.sort_values("score", ascending=False)

                # Shuffle with fixed seed for variety, then take top N
                dataset_sample = dataset_origin.sample(
                    n=min(slots_remaining * 3, len(dataset_origin)),
                    random_state=RANDOM_SEED,
                )
                dataset_sample = dataset_sample.sort_values("score", ascending=False)

                for _, row in dataset_sample.head(slots_remaining).iterrows():
                    # Get classifier confidence for this known word
                    anchored = anchor(row["word"])
                    proba = pipeline.predict_proba([anchored])[0]
                    pred = pipeline.predict([anchored])[0]
                    pred_idx = classes.index(pred)
                    conf = proba[pred_idx]

                    selected.append({
                        "word": row["word"],
                        "origin_class": origin,
                        "confidence": round(float(conf), 4),
                        "source": "dataset_etymwn",
                        "tier": 3,
                    })
                    used_words.add(row["word"])
                    slots_remaining -= 1
                    tier3_count += 1

        print(f"    Tier 3 (dataset):    {tier3_count} words")
        print(f"    Total for {origin}: {WORDS_PER_ORIGIN - slots_remaining}/{WORDS_PER_ORIGIN}")

    result_df = pd.DataFrame(selected)
    result_df = result_df.sort_values(["origin_class", "tier", "confidence"], ascending=[True, True, False])

    return result_df


# ──────────────────────────────────────────────────────────────
#  Step 5: Validate the selection
# ──────────────────────────────────────────────────────────────
def validate_selection(df: pd.DataFrame) -> bool:
    """
    Run validation checks on the final selection:
      1. Correct total count (~100)
      2. Balanced per origin (~25 each)
      3. No duplicates
      4. All words meet length requirements
    """
    print("\n[Step 5] Validating selection...")
    issues = []

    # Check total
    if len(df) != TOTAL_SHIFT_WORDS:
        issues.append(f"  ⚠ Expected {TOTAL_SHIFT_WORDS} words, got {len(df)}")

    # Check balance
    for origin in ORIGIN_CLASSES:
        count = len(df[df["origin_class"] == origin])
        if count != WORDS_PER_ORIGIN:
            issues.append(f"  ⚠ {origin}: expected {WORDS_PER_ORIGIN}, got {count}")

    # Check duplicates
    dupes = df[df.duplicated(subset=["word"], keep=False)]
    if len(dupes) > 0:
        issues.append(f"  ⚠ Found {len(dupes)} duplicate words: {dupes['word'].tolist()}")

    # Check word lengths
    bad_lengths = df[
        (df["word"].str.len() < MIN_WORD_LENGTH)
        | (df["word"].str.len() > MAX_WORD_LENGTH)
    ]
    if len(bad_lengths) > 0:
        issues.append(f"  ⚠ {len(bad_lengths)} words outside length bounds")

    if issues:
        print("  Validation issues:")
        for issue in issues:
            print(issue)
        return False
    else:
        print("  ✓ All checks passed!")
        return True


# ──────────────────────────────────────────────────────────────
#  Step 6: Save results
# ──────────────────────────────────────────────────────────────
def save_results(df: pd.DataFrame) -> None:
    """Save the selected words CSV and a detailed human-readable report."""
    print("\n[Step 6] Saving results...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── CSV output ──
    output_df = df[["word", "origin_class", "confidence", "source", "tier"]].copy()
    output_df.to_csv(SELECTED_WORDS_CSV, index=False)
    print(f"  ✓ Saved {len(output_df)} words to {SELECTED_WORDS_CSV.name}")

    # ── Human-readable report ──
    lines = []
    lines.append("=" * 70)
    lines.append("  Phase 3: Word Selection Report")
    lines.append("  Selected Words for Semantic Drift Tracking")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total words selected: {len(df)}")
    lines.append(f"Target per origin:    {WORDS_PER_ORIGIN}")
    lines.append(f"Confidence threshold: {CLASSIFIER_CONFIDENCE_THRESHOLD}")
    lines.append("")

    # Per-origin breakdown
    for origin in ORIGIN_CLASSES:
        origin_df = df[df["origin_class"] == origin]
        lines.append(f"{'─' * 70}")
        lines.append(f"  {origin} ({len(origin_df)} words)")
        lines.append(f"{'─' * 70}")

        # By tier
        for tier in sorted(origin_df["tier"].unique()):
            tier_df = origin_df[origin_df["tier"] == tier]
            tier_label = {1: "Curated Seeds", 2: "Classifier (NLTK)", 3: "Dataset (EtymWN)"}
            lines.append(f"\n  Tier {tier} — {tier_label.get(tier, 'Unknown')}:")

            for _, row in tier_df.iterrows():
                conf_bar = "█" * int(row["confidence"] * 20)
                lines.append(
                    f"    {row['word']:<20} conf={row['confidence']:.3f} {conf_bar}"
                )

        # Stats
        lines.append(f"\n  Avg confidence: {origin_df['confidence'].mean():.4f}")
        lines.append(f"  Min confidence: {origin_df['confidence'].min():.4f}")
        lines.append(f"  Max confidence: {origin_df['confidence'].max():.4f}")
        lines.append("")

    # Summary table
    lines.append(f"\n{'=' * 70}")
    lines.append("  Source Distribution")
    lines.append(f"{'=' * 70}")
    for source, count in df["source"].value_counts().items():
        lines.append(f"  {source:<20} {count:>4} words ({count / len(df) * 100:.1f}%)")

    report_text = "\n".join(lines)

    with open(SELECTION_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  ✓ Saved report to {SELECTION_REPORT.name}")

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"  Selection Summary")
    print(f"{'=' * 50}")
    for origin in ORIGIN_CLASSES:
        count = len(df[df["origin_class"] == origin])
        avg_conf = df[df["origin_class"] == origin]["confidence"].mean()
        print(f"  {origin:<12} {count:>3} words  (avg conf: {avg_conf:.3f})")
    print(f"{'─' * 50}")
    print(f"  {'TOTAL':<12} {len(df):>3} words")


# ──────────────────────────────────────────────────────────────
#  Display mode
# ──────────────────────────────────────────────────────────────
def show_selection() -> None:
    """Display the current word selection."""
    if not SELECTED_WORDS_CSV.exists():
        print("✗ No selection found. Run `python src/word_selector.py` first.")
        return

    df = pd.read_csv(SELECTED_WORDS_CSV)
    print(f"\n{'=' * 65}")
    print(f"  Selected Words for Semantic Drift Tracking ({len(df)} words)")
    print(f"{'=' * 65}")

    for origin in ORIGIN_CLASSES:
        origin_df = df[df["origin_class"] == origin].sort_values("tier")
        print(f"\n  ── {origin} ({len(origin_df)} words) ──")
        for _, row in origin_df.iterrows():
            tier_symbol = {1: "★", 2: "●", 3: "○"}.get(row["tier"], "?")
            print(f"    {tier_symbol} {row['word']:<20} conf={row['confidence']:.3f}  [{row['source']}]")

    print(f"\n  Legend: ★ = curated seed  ● = classifier pick  ○ = dataset word")


def show_stats() -> None:
    """Show detailed statistics about the selection."""
    if not SELECTED_WORDS_CSV.exists():
        print("✗ No selection found. Run `python src/word_selector.py` first.")
        return

    df = pd.read_csv(SELECTED_WORDS_CSV)

    print(f"\n{'=' * 55}")
    print(f"  Selection Statistics")
    print(f"{'=' * 55}")
    print(f"\n  Total words: {len(df)}")
    print(f"\n  By origin class:")
    for origin in ORIGIN_CLASSES:
        odf = df[df["origin_class"] == origin]
        print(f"    {origin:<12} {len(odf):>3}  avg_conf={odf['confidence'].mean():.3f}  "
              f"avg_len={odf['word'].str.len().mean():.1f}")

    print(f"\n  By source:")
    for source, count in df["source"].value_counts().items():
        print(f"    {source:<20} {count:>3} ({count / len(df) * 100:.0f}%)")

    print(f"\n  By tier:")
    for tier in sorted(df["tier"].unique()):
        count = len(df[df["tier"] == tier])
        label = {1: "Curated Seeds", 2: "Classifier (NLTK)", 3: "Dataset (EtymWN)"}
        print(f"    Tier {tier} ({label.get(tier, '?')}): {count} words")

    print(f"\n  Word length distribution:")
    lengths = df["word"].str.len()
    print(f"    Min: {lengths.min()}  Max: {lengths.max()}  "
          f"Mean: {lengths.mean():.1f}  Median: {lengths.median():.0f}")

    print(f"\n  Confidence distribution:")
    print(f"    Min: {df['confidence'].min():.4f}  Max: {df['confidence'].max():.4f}  "
          f"Mean: {df['confidence'].mean():.4f}  Median: {df['confidence'].median():.4f}")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Select ~100 words for semantic drift tracking"
    )
    parser.add_argument("--show", action="store_true",
                        help="Display current word selection")
    parser.add_argument("--stats", action="store_true",
                        help="Show detailed statistics")
    args = parser.parse_args()

    if args.show:
        show_selection()
        return

    if args.stats:
        show_stats()
        return

    print("=" * 60)
    print("  Phase 3: Word Selection for Semantic Shift")
    print("  Goal: Select 100 words (25 per origin) for drift tracking")
    print("=" * 60)

    # Step 1: Load classifier
    pipeline, classes = load_classifier()

    # Step 2: Load candidate words + frequency data + stopwords
    nltk_candidates, word_freq, stopwords = load_candidate_words()
    dataset_df = load_dataset_words()
    known_words = set(dataset_df["word"].tolist())

    # Step 3: Classify NLTK candidates (frequency + stopword filtered)
    classifier_df = classify_candidates(
        pipeline, classes, nltk_candidates, known_words, word_freq, stopwords
    )

    # Step 4: Build balanced selection
    selection_df = select_words(pipeline, classes, classifier_df, dataset_df)

    # Step 5: Validate
    is_valid = validate_selection(selection_df)

    # Step 6: Save
    save_results(selection_df)

    print(f"\n{'=' * 60}")
    print(f"  Phase 3 Complete!")
    print(f"  {len(selection_df)} words selected for semantic drift tracking")
    print(f"  Validation: {'✓ PASSED' if is_valid else '⚠ ISSUES FOUND'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
