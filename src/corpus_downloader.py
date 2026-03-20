"""
Corpus Downloader (Phase 4 — Steps 4.1 & 4.2)
================================================
Downloads and assembles two diachronic corpora for semantic drift analysis:

  Period A (Historical, 1800–1900):
    - Source: Project Gutenberg via direct HTTP download
    - Uses a curated list of well-known 19th-century English books
    - Plus randomly sampled books from Gutenberg's mirror
    - Strips boilerplate headers/footers
    - Target: ~5–10M tokens

  Period B (Modern, 2000–2020):
    - Source: English Wikipedia via HuggingFace datasets (streaming)
    - Represents contemporary standard English
    - Target: match historical corpus size (±15%)

Usage:
    python src/corpus_downloader.py                # Download both
    python src/corpus_downloader.py --historical   # Download historical only
    python src/corpus_downloader.py --modern       # Download modern only
    python src/corpus_downloader.py --status       # Check current corpus status
"""

import sys
import time
import json
import re
import argparse
import warnings
import urllib.request
import urllib.error
from pathlib import Path

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    CORPORA_RAW_DIR, CORPORA_PROCESSED_DIR,
    GUTENBERG_RAW, GUTENBERG_BOOKS_TARGET,
    WIKI_RAW, WIKI_ARTICLES_TARGET,
    TARGET_TOKENS,
    PERIOD_A, PERIOD_B,
)


# ──────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    """Quick whitespace-based token count."""
    return len(text.split())


def format_tokens(n: int) -> str:
    """Human-readable token count."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg header and footer boilerplate.
    
    Gutenberg texts have standardized markers:
      - Header ends with: "*** START OF THE PROJECT GUTENBERG EBOOK <title> ***"
      - Footer begins with: "*** END OF THE PROJECT GUTENBERG EBOOK <title> ***"
    
    We strip everything outside these markers.
    """
    # Find START marker
    start_markers = [
        r"\*\*\* ?START OF (?:THE |THIS )?PROJECT GUTENBERG",
        r"\*\*\*START OF (?:THE |THIS )?PROJECT GUTENBERG",
        r"End of (?:the )?Project Gutenberg Header",
    ]
    start_pos = 0
    for pattern in start_markers:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Skip past the marker line
            line_end = text.find("\n", match.end())
            if line_end != -1:
                start_pos = line_end + 1
            break

    # Find END marker
    end_markers = [
        r"\*\*\* ?END OF (?:THE |THIS )?PROJECT GUTENBERG",
        r"\*\*\*END OF (?:THE |THIS )?PROJECT GUTENBERG",
        r"End of (?:the )?Project Gutenberg",
    ]
    end_pos = len(text)
    for pattern in end_markers:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            end_pos = match.start()
            break

    return text[start_pos:end_pos].strip()


# ──────────────────────────────────────────────────────────────
#  Curated list of Gutenberg book IDs (19th-century English)
# ──────────────────────────────────────────────────────────────
# These are well-known English-language works from 1800–1900.
# They cover diverse genres: novels, essays, science, philosophy,
# adventure, children's literature, and social commentary.
CURATED_GUTENBERG_IDS = [
    # ── Jane Austen ──
    1342,   # Pride and Prejudice
    158,    # Emma
    161,    # Sense and Sensibility
    105,    # Persuasion
    121,    # Northanger Abbey
    141,    # Mansfield Park

    # ── Charles Dickens ──
    98,     # A Tale of Two Cities
    1400,   # Great Expectations
    730,    # Oliver Twist
    766,    # David Copperfield
    580,    # The Pickwick Papers
    564,    # The Old Curiosity Shop
    883,    # Bleak House
    821,    # Dombey and Son
    967,    # Our Mutual Friend
    700,    # The Cricket on the Hearth
    46,     # A Christmas Carol

    # ── Charlotte & Emily Brontë ──
    1260,   # Jane Eyre
    768,    # Wuthering Heights

    # ── George Eliot ──
    145,    # Middlemarch
    550,    # Silas Marner
    6688,   # The Mill on the Floss

    # ── Thomas Hardy ──
    110,    # Tess of the d'Urbervilles
    153,    # Jude the Obscure
    17500,  # Far from the Madding Crowd

    # ── Mark Twain ──
    76,     # Adventures of Tom Sawyer
    74,     # The Adventures of Tom Sawyer (alt)
    86,     # A Connecticut Yankee
    3176,   # The Prince and the Pauper
    245,    # Roughing It

    # ── Herman Melville ──
    2701,   # Moby Dick
    15,     # Bartleby, the Scrivener

    # ── Mary Shelley / Gothic ──
    84,     # Frankenstein
    345,    # Dracula (Bram Stoker)
    42,     # The Strange Case of Dr Jekyll and Mr Hyde

    # ── Lewis Carroll ──
    11,     # Alice's Adventures in Wonderland
    12,     # Through the Looking-Glass

    # ── Louisa May Alcott ──
    514,    # Little Women
    37106,  # An Old-Fashioned Girl

    # ── Nathaniel Hawthorne ──
    33,     # The Scarlet Letter
    77,     # The House of the Seven Gables

    # ── Henry James ──
    209,    # The Turn of the Screw
    432,    # The Portrait of a Lady
    7118,   # The Wings of the Dove

    # ── Oscar Wilde ──
    174,    # The Picture of Dorian Gray
    885,    # An Ideal Husband

    # ── Jules Verne (English translations) ──
    103,    # Around the World in 80 Days
    164,    # Twenty Thousand Leagues
    18857,  # Journey to the Center of the Earth

    # ── HG Wells ──
    35,     # The Time Machine
    36,     # The War of the Worlds
    5230,   # The Invisible Man

    # ── Arthur Conan Doyle ──
    1661,   # Adventures of Sherlock Holmes
    233,    # The Hound of the Baskervilles
    244,    # A Study in Scarlet
    2852,   # The Sign of the Four

    # ── Robert Louis Stevenson ──
    120,    # Treasure Island
    43,     # The Strange Case of Dr Jekyll and Mr Hyde (alt)

    # ── Rudyard Kipling ──
    35997,  # The Jungle Book
    236,    # The Second Jungle Book

    # ── Non-fiction / Philosophy / Science ──
    2009,   # On the Origin of Species (Darwin)
    4280,   # The Descent of Man (Darwin)
    10615,  # Walden (Thoreau)
    3207,   # Leviathan (Hobbes, but Gutenberg text from 19th-c edition)
    1497,   # Republic (Plato, but 19th-c translation by Jowett)
    4300,   # Ulysses (James Joyce, though early 20th century)
    5827,   # The Problems of Philosophy (Bertrand Russell)
    36,     # The War of the Worlds

    # ── Edgar Allan Poe ──
    932,    # The Fall of the House of Usher
    2147,   # Tales of Mystery and Imagination

    # ── Other notable 19th-century works ──
    1952,   # The Yellow Wallpaper (Gilman)
    834,    # The Count of Monte Cristo excerpt
    1232,   # The Prince (Machiavelli, 19th-c translation)
    2600,   # War and Peace (Tolstoy, English translation)
    1399,   # Anna Karenina (Tolstoy, English translation)
    135,    # Les Misérables (Hugo, English translation)
    5200,   # Metamorphosis (Kafka, though just post-1900)
    174,    # The Picture of Dorian Gray (alt)
    2500,   # Siddhartha (alt)
    6130,   # The Iliad (Pope translation)
    3600,   # The Odyssey (Pope/Butler translation)

    # ── Additional gap-fillers for token count ──
    1023,   # Bleak House (alt)
    27827,  # The Kama Sutra (Burton translation)
    996,    # Don Quixote (Cervantes, English trans)
    76,     # Tom Sawyer (alt)
    2554,   # Crime and Punishment (English trans)
    1184,   # The Count of Monte Cristo
    16643,  # The Brothers Karamazov
    28054,  # The Brothers Karamazov (alt)
]


def download_gutenberg_text(book_id: int) -> str | None:
    """
    Download a single book from Project Gutenberg's mirror.
    
    Tries multiple URL formats with retry logic.
    Catches ALL network errors to prevent crashes mid-download.
    """
    url_patterns = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]

    for url in url_patterns:
        for attempt in range(2):  # Retry once per URL
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "ETYMO-Research/1.0 (Academic Project)"}
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    raw = response.read()
                    try:
                        text = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        text = raw.decode("latin-1")
                    return text
            except Exception:
                # Catches: URLError, HTTPError, TimeoutError, IncompleteRead,
                # ConnectionResetError, socket.timeout, and anything else
                if attempt == 0:
                    time.sleep(1)  # Brief pause before retry
                continue

    return None


# ──────────────────────────────────────────────────────────────
#  Step 4.1: Download Historical Corpus (Project Gutenberg)
# ──────────────────────────────────────────────────────────────
def download_gutenberg_corpus() -> Path:
    """
    Download English-language books from Project Gutenberg to build
    a historical corpus (~1800–1900).

    Strategy:
      1. Download from curated list of well-known 19th-century books
      2. If we need more tokens, try additional IDs from Gutenberg
      3. Strip boilerplate, apply basic quality filters
      4. Save merged text file + metadata

    Returns:
        Path to the merged raw text file.
    """
    print("\n[Step 4.1] Downloading Historical Corpus (Project Gutenberg)")
    print(f"  Period: {PERIOD_A['start']}–{PERIOD_A['end']}")
    print(f"  Target: {format_tokens(TARGET_TOKENS)} tokens")

    GUTENBERG_RAW.mkdir(parents=True, exist_ok=True)

    # Check cache
    merged_file = GUTENBERG_RAW / "gutenberg_merged.txt"
    if merged_file.exists():
        existing_tokens = count_tokens(
            merged_file.read_text(encoding="utf-8", errors="ignore")
        )
        if existing_tokens >= TARGET_TOKENS * 0.85:
            print(f"  ✓ Already have {format_tokens(existing_tokens)} tokens in cache")
            return merged_file

    # ── Download books ──
    print(f"\n  Downloading from curated list ({len(CURATED_GUTENBERG_IDS)} book IDs)...")
    print(f"  (This downloads directly from gutenberg.org — may take 10–20 minutes)")

    # Deduplicate
    unique_ids = list(dict.fromkeys(CURATED_GUTENBERG_IDS))

    total_tokens = 0
    books_downloaded = 0
    books_failed = 0
    all_texts = []

    for i, book_id in enumerate(unique_ids):
        if total_tokens >= TARGET_TOKENS:
            break

        text = download_gutenberg_text(book_id)
        if text is None:
            books_failed += 1
            continue

        # Strip boilerplate
        clean = strip_gutenberg_boilerplate(text)
        tokens = count_tokens(clean)

        # Quality filters
        if tokens < 2000:
            books_failed += 1
            continue
        if tokens > 600000:
            # Very long books: take first 600K tokens worth
            words = clean.split()[:600000]
            clean = " ".join(words)
            tokens = 600000

        # Quick language check
        ascii_ratio = sum(1 for c in clean[:2000] if ord(c) < 128) / min(2000, len(clean))
        if ascii_ratio < 0.8:
            books_failed += 1
            continue

        all_texts.append(clean)
        total_tokens += tokens
        books_downloaded += 1

        if books_downloaded % 5 == 0 or books_downloaded == 1:
            print(f"    [{books_downloaded:>3}] ID {book_id:<6} │ "
                  f"{tokens:>7,} tokens │ "
                  f"Total: {format_tokens(total_tokens)}")

        # Small delay to be polite to Gutenberg servers
        time.sleep(0.5)

    # ── If we still need more, try additional random IDs ──
    if total_tokens < TARGET_TOKENS * 0.85:
        print(f"\n  Need more tokens ({format_tokens(total_tokens)} < "
              f"{format_tokens(int(TARGET_TOKENS * 0.85))})")
        print(f"  Trying additional Gutenberg IDs...")

        import random
        random.seed(42)
        extra_ids = list(range(100, 40000))
        random.shuffle(extra_ids)
        already_tried = set(unique_ids)

        for book_id in extra_ids:
            if total_tokens >= TARGET_TOKENS:
                break
            if book_id in already_tried:
                continue

            already_tried.add(book_id)
            text = download_gutenberg_text(book_id)
            if text is None:
                continue

            clean = strip_gutenberg_boilerplate(text)
            tokens = count_tokens(clean)

            if tokens < 5000:
                continue

            ascii_ratio = sum(1 for c in clean[:2000] if ord(c) < 128) / min(2000, len(clean))
            if ascii_ratio < 0.85:
                continue

            all_texts.append(clean)
            total_tokens += tokens
            books_downloaded += 1

            if books_downloaded % 10 == 0:
                print(f"    [{books_downloaded:>3}] ID {book_id:<6} │ "
                      f"{tokens:>7,} tokens │ "
                      f"Total: {format_tokens(total_tokens)}")

            time.sleep(0.5)

    print(f"\n  ✓ Downloaded {books_downloaded} books")
    print(f"  ✓ Total raw tokens: {format_tokens(total_tokens)}")
    print(f"  ✗ Failed/skipped: {books_failed}")

    # ── Merge and save ──
    print(f"\n  Merging texts...")
    merged_text = "\n\n".join(all_texts)
    merged_file.write_text(merged_text, encoding="utf-8")
    size_mb = merged_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {merged_file.name} ({size_mb:.1f} MB)")

    # Save metadata
    meta = {
        "source": "Project Gutenberg (direct download)",
        "period": f"{PERIOD_A['start']}–{PERIOD_A['end']}",
        "books_downloaded": books_downloaded,
        "books_failed": books_failed,
        "total_tokens": total_tokens,
        "file_size_mb": round(size_mb, 2),
    }
    with open(GUTENBERG_RAW / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return merged_file


# ──────────────────────────────────────────────────────────────
#  Step 4.2: Download Modern Corpus (Wikipedia)
# ──────────────────────────────────────────────────────────────
def download_wikipedia_corpus() -> Path:
    """
    Download modern English text from Wikipedia via HuggingFace datasets.

    Strategy:
      1. Stream Wikipedia articles (no full dump needed)
      2. Skip stubs, disambiguation, and list pages
      3. Accumulate until target token count
      4. Save merged text file + metadata

    Returns:
        Path to the merged raw text file.
    """
    print("\n[Step 4.2] Downloading Modern Corpus (Wikipedia)")
    print(f"  Period: {PERIOD_B['start']}–{PERIOD_B['end']}")
    print(f"  Target: {format_tokens(TARGET_TOKENS)} tokens")

    WIKI_RAW.mkdir(parents=True, exist_ok=True)

    # Check cache
    merged_file = WIKI_RAW / "wikipedia_merged.txt"
    if merged_file.exists():
        existing_tokens = count_tokens(
            merged_file.read_text(encoding="utf-8", errors="ignore")
        )
        if existing_tokens >= TARGET_TOKENS * 0.85:
            print(f"  ✓ Already have {format_tokens(existing_tokens)} tokens in cache")
            return merged_file

    from datasets import load_dataset

    # ── Load Wikipedia ──
    print("\n  Loading Wikipedia dataset (streaming mode)...")
    print("  This downloads articles on demand — no full dump needed.")

    try:
        dataset = load_dataset(
            "wikipedia",
            "20220301.en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  ⚠ Primary Wikipedia load failed: {e}")
        print("  Trying alternative configuration...")
        try:
            dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as e2:
            print(f"  ✗ Wikipedia loading failed: {e2}")
            print("  You may need to run: huggingface-cli login")
            sys.exit(1)

    # ── Stream articles ──
    print(f"\n  Streaming articles (target: {format_tokens(TARGET_TOKENS)} tokens)...")

    total_tokens = 0
    articles_used = 0
    articles_skipped = 0
    all_texts = []

    for article in dataset:
        if total_tokens >= TARGET_TOKENS:
            break
        if articles_used >= WIKI_ARTICLES_TARGET:
            break

        text = article.get("text", "")
        if not text:
            continue

        # Skip short stubs
        tokens = count_tokens(text)
        if tokens < 100:
            articles_skipped += 1
            continue

        # Skip disambiguation pages
        lower_start = text[:500].lower()
        if "may refer to" in lower_start or "disambiguation" in lower_start:
            articles_skipped += 1
            continue

        # Skip list-heavy articles
        newlines = text.count("\n")
        bullets = text.count("\n*") + text.count("\n-")
        if newlines > 0 and bullets / max(newlines, 1) > 0.4:
            articles_skipped += 1
            continue

        all_texts.append(text)
        total_tokens += tokens
        articles_used += 1

        if articles_used % 2000 == 0:
            print(f"    {articles_used:>6,} articles │ "
                  f"{format_tokens(total_tokens)} tokens")

    print(f"\n  ✓ Used {articles_used:,} articles")
    print(f"  ✓ Total tokens: {format_tokens(total_tokens)}")
    print(f"  ○ Skipped {articles_skipped:,} stubs/disambiguation")

    # ── Save ──
    print(f"\n  Merging texts...")
    merged_text = "\n\n".join(all_texts)
    merged_file.write_text(merged_text, encoding="utf-8")
    size_mb = merged_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {merged_file.name} ({size_mb:.1f} MB)")

    meta = {
        "source": "English Wikipedia (HuggingFace datasets)",
        "period": f"{PERIOD_B['start']}–{PERIOD_B['end']}",
        "articles_used": articles_used,
        "articles_skipped": articles_skipped,
        "total_tokens": total_tokens,
        "file_size_mb": round(size_mb, 2),
    }
    with open(WIKI_RAW / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return merged_file


# ──────────────────────────────────────────────────────────────
#  Status check
# ──────────────────────────────────────────────────────────────
def check_status():
    """Report on current corpus download state."""
    print(f"\n{'=' * 55}")
    print(f"  Corpus Download Status")
    print(f"{'=' * 55}")

    for label, raw_dir, fname in [
        ("Historical (Gutenberg)", GUTENBERG_RAW, "gutenberg_merged.txt"),
        ("Modern (Wikipedia)", WIKI_RAW, "wikipedia_merged.txt"),
    ]:
        merged = raw_dir / fname
        meta_file = raw_dir / "metadata.json"

        print(f"\n  ── {label} ──")
        if merged.exists():
            tokens = count_tokens(merged.read_text(encoding="utf-8", errors="ignore"))
            size_mb = merged.stat().st_size / (1024 * 1024)
            status = "✓ READY" if tokens >= TARGET_TOKENS * 0.85 else "⚠ PARTIAL"
            print(f"    Status: {status}")
            print(f"    Tokens: {format_tokens(tokens)}")
            print(f"    Size:   {size_mb:.1f} MB")
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                for k, v in meta.items():
                    if k not in ("total_tokens", "file_size_mb"):
                        print(f"    {k}: {v}")
        else:
            print(f"    Status: ✗ NOT DOWNLOADED")

    print(f"\n  Target per corpus: {format_tokens(TARGET_TOKENS)} tokens")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Download diachronic corpora"
    )
    parser.add_argument("--historical", action="store_true",
                        help="Download historical corpus only")
    parser.add_argument("--modern", action="store_true",
                        help="Download modern corpus only")
    parser.add_argument("--status", action="store_true",
                        help="Show current download status")
    args = parser.parse_args()

    if args.status:
        check_status()
        return

    print("=" * 60)
    print("  Phase 4: Diachronic Corpus Download")
    print("=" * 60)

    t_start = time.time()

    do_hist = args.historical or (not args.historical and not args.modern)
    do_mod = args.modern or (not args.historical and not args.modern)

    if do_hist:
        download_gutenberg_corpus()

    if do_mod:
        download_wikipedia_corpus()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Downloads complete! ({elapsed / 60:.1f} minutes)")
    print(f"  Next: python src/corpus_preprocessor.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
