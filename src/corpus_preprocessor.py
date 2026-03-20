"""
Corpus Preprocessor (Phase 4 — Step 4.3)
==========================================
Applies IDENTICAL preprocessing to both the historical (Gutenberg) and
modern (Wikipedia) corpora, ensuring that any measured semantic drift
reflects genuine language change, not preprocessing artifacts.

Pipeline (applied identically to both):
  1. Lowercase everything
  2. Sentence segmentation (NLTK sent_tokenize)
  3. Word tokenization (NLTK word_tokenize)
  4. Remove punctuation, numbers, single-character tokens
  5. Do NOT stem or lemmatize (we need original word forms)
  6. Remove sentences shorter than 5 tokens
  7. Log token and sentence counts for comparability

Output format: One sentence per line, space-separated tokens.

Usage:
    python src/corpus_preprocessor.py                # Process both corpora
    python src/corpus_preprocessor.py --historical   # Process historical only
    python src/corpus_preprocessor.py --modern       # Process modern only
    python src/corpus_preprocessor.py --stats        # Show corpus statistics
    python src/corpus_preprocessor.py --verify       # Verify selected words in both
"""

import sys
import time
import json
import re
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    CORPORA_RAW_DIR, CORPORA_PROCESSED_DIR, RESULTS_DIR,
    GUTENBERG_RAW, WIKI_RAW,
    CORPUS_OLD, CORPUS_NEW, CORPUS_STATS,
    TARGET_TOKENS, TOKEN_TOLERANCE,
    MIN_SENTENCE_TOKENS, MIN_TOKEN_LENGTH,
    SELECTED_WORDS_CSV,
)


# ──────────────────────────────────────────────────────────────
#  The ONE preprocessing function (shared by both corpora)
# ──────────────────────────────────────────────────────────────
def preprocess_text(
    raw_text: str,
    corpus_name: str,
    max_tokens: int | None = None,
) -> tuple[list[str], dict]:
    """
    Preprocess raw text into clean sentences for Word2Vec training.
    
    CRITICAL: This exact same function is called for BOTH corpora.
    Any change here affects both identically — which is the whole point.
    
    Args:
        raw_text: The raw text to process (may be very large)
        corpus_name: Label for logging ("historical" or "modern")
        max_tokens: Optional cap on total tokens (for balancing corpus sizes)
    
    Returns:
        sentences: List of cleaned sentences (each is space-separated tokens)
        stats: Dict of statistics about the preprocessing
    """
    import nltk
    nltk.download("punkt_tab", quiet=True)

    print(f"\n  ── Preprocessing: {corpus_name} ──")

    # Track statistics
    stats = {
        "corpus": corpus_name,
        "raw_characters": len(raw_text),
    }

    # ── Step 1: Lowercase ──
    print(f"    [1/6] Lowercasing...")
    text = raw_text.lower()

    # ── Step 2: Sentence segmentation ──
    print(f"    [2/6] Segmenting into sentences...")
    # Process in chunks to avoid memory issues with very large texts
    chunk_size = 500_000  # characters per chunk
    raw_sentences = []
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        # Don't split in the middle of a sentence — extend to next period
        if i + chunk_size < len(text):
            # Find the last sentence-ending punctuation
            for j in range(min(1000, len(text) - i - chunk_size)):
                if chunk_size + j < len(text) - i and text[i + chunk_size + j] in '.!?\n':
                    chunk = text[i : i + chunk_size + j + 1]
                    break
        
        sents = nltk.sent_tokenize(chunk)
        raw_sentences.extend(sents)

    stats["raw_sentences"] = len(raw_sentences)
    print(f"      {len(raw_sentences):,} raw sentences")

    # ── Step 3 & 4: Tokenize, clean, filter ──
    print(f"    [3/6] Tokenizing and cleaning...")
    
    # Compile regex patterns for speed (called millions of times)
    alpha_pattern = re.compile(r'^[a-z]+$')
    
    clean_sentences = []
    total_tokens = 0
    total_removed_tokens = 0
    
    for sent in raw_sentences:
        # Tokenize (NLTK word_tokenize handles contractions, etc.)
        tokens = nltk.word_tokenize(sent)
        
        # Filter tokens:
        #   - Keep only alphabetic tokens (removes punctuation, numbers)
        #   - Minimum length (removes single-character tokens like 'i', 'a' etc.
        #     ... actually 'i' and 'a' are valid, so min_length=2 removes only
        #     truly useless single chars from tokenization artifacts)
        clean_tokens = []
        for tok in tokens:
            if alpha_pattern.match(tok) and len(tok) >= MIN_TOKEN_LENGTH:
                clean_tokens.append(tok)
            else:
                total_removed_tokens += 1
        
        # ── Step 5: Skip short sentences ──
        if len(clean_tokens) < MIN_SENTENCE_TOKENS:
            continue
        
        clean_sentences.append(" ".join(clean_tokens))
        total_tokens += len(clean_tokens)
        
        # ── Check token cap ──
        if max_tokens and total_tokens >= max_tokens:
            break

    stats["clean_sentences"] = len(clean_sentences)
    stats["clean_tokens"] = total_tokens
    stats["removed_tokens"] = total_removed_tokens
    stats["avg_sentence_length"] = round(total_tokens / max(len(clean_sentences), 1), 1)

    print(f"      {len(clean_sentences):,} clean sentences")
    print(f"      {total_tokens:,} clean tokens")
    print(f"      {total_removed_tokens:,} tokens removed (punctuation/numbers/short)")
    print(f"      Avg sentence length: {stats['avg_sentence_length']} tokens")

    # ── Build vocabulary stats ──
    print(f"    [4/6] Computing vocabulary...")
    vocab = Counter()
    for sent in clean_sentences:
        for tok in sent.split():
            vocab[tok] += 1

    stats["vocabulary_size"] = len(vocab)
    stats["vocabulary_min10"] = sum(1 for _, c in vocab.items() if c >= 10)
    stats["top_20_words"] = [w for w, _ in vocab.most_common(20)]

    print(f"      Total vocabulary: {len(vocab):,} unique words")
    print(f"      Vocabulary (min_count≥10): {stats['vocabulary_min10']:,} words")
    print(f"      Top 20: {', '.join(stats['top_20_words'][:10])}...")

    return clean_sentences, stats


# ──────────────────────────────────────────────────────────────
#  Process a corpus end-to-end
# ──────────────────────────────────────────────────────────────
def process_corpus(
    raw_dir: Path,
    merged_filename: str,
    output_path: Path,
    corpus_name: str,
    max_tokens: int | None = None,
) -> dict:
    """
    Load raw text → preprocess → save clean output.
    
    Returns stats dict.
    """
    merged_file = raw_dir / merged_filename

    if not merged_file.exists():
        print(f"\n  ✗ Raw corpus not found at {merged_file}")
        print(f"    Run `python src/corpus_downloader.py` first.")
        sys.exit(1)

    print(f"\n  Loading raw text from {merged_file.name}...")
    raw_text = merged_file.read_text(encoding="utf-8", errors="ignore")
    raw_tokens = len(raw_text.split())
    print(f"    Raw: {raw_tokens:,} tokens, {len(raw_text):,} characters")

    # Preprocess
    sentences, stats = preprocess_text(raw_text, corpus_name, max_tokens)

    # Save
    CORPORA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sentences), encoding="utf-8")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    stats["output_file"] = str(output_path.name)
    stats["output_size_mb"] = round(size_mb, 2)

    print(f"\n    [5/6] Saved to {output_path.name} ({size_mb:.1f} MB)")

    return stats


# ──────────────────────────────────────────────────────────────
#  Verify selected words exist in both corpora
# ──────────────────────────────────────────────────────────────
def verify_selected_words(stats_old: dict | None = None, stats_new: dict | None = None):
    """
    Check what fraction of our 100 selected words appear in each corpus.
    Words that don't appear in BOTH corpora can't be used for drift analysis.
    """
    print(f"\n    [6/6] Verifying selected words in both corpora...")

    if not SELECTED_WORDS_CSV.exists():
        print(f"      ⚠ No selected_words.csv found, skipping verification")
        return

    selected = pd.read_csv(SELECTED_WORDS_CSV)
    selected_words = set(selected["word"].tolist())

    # Build vocab from processed corpora
    results = {}
    for label, corpus_path in [("historical", CORPUS_OLD), ("modern", CORPUS_NEW)]:
        if not corpus_path.exists():
            print(f"      ⚠ {corpus_path.name} not found, skipping")
            continue

        vocab = Counter()
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                for tok in line.strip().split():
                    if tok in selected_words:
                        vocab[tok] += 1

        found = set(vocab.keys())
        missing = selected_words - found
        results[label] = {
            "found": len(found),
            "missing": len(missing),
            "missing_words": sorted(missing),
            "coverage": round(len(found) / len(selected_words) * 100, 1),
        }

        print(f"\n      {label.capitalize()} corpus:")
        print(f"        Found: {len(found)}/{len(selected_words)} "
              f"({results[label]['coverage']}%)")
        if missing:
            print(f"        Missing: {', '.join(sorted(missing)[:20])}")
            if len(missing) > 20:
                print(f"        ... and {len(missing) - 20} more")

    # Words in BOTH corpora
    if len(results) == 2:
        found_both = (selected_words
                      - set(results["historical"].get("missing_words", []))
                      - set(results["modern"].get("missing_words", [])))
        print(f"\n      Words in BOTH corpora: {len(found_both)}/{len(selected_words)}")
        if len(found_both) < len(selected_words):
            all_missing = (set(results["historical"].get("missing_words", []))
                          | set(results["modern"].get("missing_words", [])))
            print(f"      ⚠ {len(all_missing)} words missing from at least one corpus:")
            print(f"        {', '.join(sorted(all_missing))}")

    return results


# ──────────────────────────────────────────────────────────────
#  Write comparison report
# ──────────────────────────────────────────────────────────────
def write_stats_report(stats_old: dict, stats_new: dict):
    """Write a detailed comparison report of both corpora."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("  Phase 4: Corpus Comparison Report")
    lines.append("=" * 70)
    lines.append("")

    # Side-by-side comparison
    lines.append(f"{'Metric':<35} {'Historical':>15} {'Modern':>15}")
    lines.append("─" * 70)

    metrics = [
        ("Raw characters", "raw_characters", "{:,}"),
        ("Raw sentences", "raw_sentences", "{:,}"),
        ("Clean sentences", "clean_sentences", "{:,}"),
        ("Clean tokens", "clean_tokens", "{:,}"),
        ("Removed tokens", "removed_tokens", "{:,}"),
        ("Avg sentence length", "avg_sentence_length", "{:.1f}"),
        ("Vocabulary (total)", "vocabulary_size", "{:,}"),
        ("Vocabulary (min_count≥10)", "vocabulary_min10", "{:,}"),
        ("Output file size (MB)", "output_size_mb", "{:.1f}"),
    ]

    for label, key, fmt in metrics:
        val_old = stats_old.get(key, "N/A")
        val_new = stats_new.get(key, "N/A")
        if val_old != "N/A":
            val_old_str = fmt.format(val_old)
        else:
            val_old_str = "N/A"
        if val_new != "N/A":
            val_new_str = fmt.format(val_new)
        else:
            val_new_str = "N/A"
        lines.append(f"  {label:<33} {val_old_str:>15} {val_new_str:>15}")

    # Token balance check
    lines.append("")
    lines.append("─" * 70)
    old_tokens = stats_old.get("clean_tokens", 0)
    new_tokens = stats_new.get("clean_tokens", 0)
    if old_tokens > 0 and new_tokens > 0:
        ratio = new_tokens / old_tokens
        balance_ok = abs(ratio - 1.0) <= TOKEN_TOLERANCE
        lines.append(f"  Token ratio (modern/historical): {ratio:.3f}")
        lines.append(f"  Balance: {'✓ GOOD' if balance_ok else '⚠ IMBALANCED'} "
                     f"(target: 1.0 ± {TOKEN_TOLERANCE})")

    report_text = "\n".join(lines)

    CORPUS_STATS.write_text(report_text, encoding="utf-8")
    print(f"\n  ✓ Corpus comparison report saved to {CORPUS_STATS.name}")
    print(f"\n{report_text}")


# ──────────────────────────────────────────────────────────────
#  Show stats mode
# ──────────────────────────────────────────────────────────────
def show_stats():
    """Display stats for existing processed corpora."""
    print(f"\n{'=' * 60}")
    print(f"  Processed Corpus Statistics")
    print(f"{'=' * 60}")

    for label, path in [("Historical", CORPUS_OLD), ("Modern", CORPUS_NEW)]:
        print(f"\n  ── {label} ({path.name}) ──")
        if not path.exists():
            print(f"    Status: ✗ NOT PROCESSED")
            continue

        # Count lines and tokens
        total_lines = 0
        total_tokens = 0
        vocab = Counter()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                tokens = line.strip().split()
                total_tokens += len(tokens)
                for tok in tokens:
                    vocab[tok] += 1

        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"    Status: ✓ READY")
        print(f"    Sentences: {total_lines:,}")
        print(f"    Tokens: {total_tokens:,}")
        print(f"    Vocabulary: {len(vocab):,} unique words")
        print(f"    Vocab (min_count≥10): {sum(1 for _, c in vocab.items() if c >= 10):,}")
        print(f"    Size: {size_mb:.1f} MB")
        print(f"    Top 10: {', '.join(w for w, _ in vocab.most_common(10))}")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 4.3: Preprocess both corpora with identical pipeline"
    )
    parser.add_argument("--historical", action="store_true",
                        help="Process historical corpus only")
    parser.add_argument("--modern", action="store_true",
                        help="Process modern corpus only")
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics for processed corpora")
    parser.add_argument("--verify", action="store_true",
                        help="Verify selected words appear in both corpora")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.verify:
        verify_selected_words()
        return

    print("=" * 60)
    print("  Phase 4.3: Corpus Preprocessing")
    print("  Applying IDENTICAL pipeline to both corpora")
    print("=" * 60)

    t_start = time.time()

    process_old = args.historical or (not args.historical and not args.modern)
    process_new = args.modern or (not args.historical and not args.modern)

    stats_old = None
    stats_new = None

    # ── Process historical corpus ──
    if process_old:
        stats_old = process_corpus(
            raw_dir=GUTENBERG_RAW,
            merged_filename="gutenberg_merged.txt",
            output_path=CORPUS_OLD,
            corpus_name="Historical (1800–1900)",
        )

    # ── Process modern corpus ──
    # If we have the historical token count, cap modern to match (±tolerance)
    if process_new:
        max_tokens_new = None
        if stats_old:
            # Match historical corpus size for fair comparison
            max_tokens_new = int(stats_old["clean_tokens"] * (1 + TOKEN_TOLERANCE))
            print(f"\n  ℹ Capping modern corpus to {max_tokens_new:,} tokens "
                  f"(matching historical ± {TOKEN_TOLERANCE*100:.0f}%)")

        stats_new = process_corpus(
            raw_dir=WIKI_RAW,
            merged_filename="wikipedia_merged.txt",
            output_path=CORPUS_NEW,
            corpus_name="Modern (2000–2020)",
            max_tokens=max_tokens_new,
        )

    # ── Comparison report ──
    if stats_old and stats_new:
        write_stats_report(stats_old, stats_new)

    # ── Verify selected words ──
    verify_selected_words(stats_old, stats_new)

    elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"  Phase 4.3 Complete!")
    print(f"  Time: {elapsed:.1f} seconds")
    if stats_old:
        print(f"  Historical: {stats_old['clean_tokens']:,} tokens → {CORPUS_OLD.name}")
    if stats_new:
        print(f"  Modern:     {stats_new['clean_tokens']:,} tokens → {CORPUS_NEW.name}")
    print(f"  Next: Run `python src/train_embeddings.py` (Phase 5)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
