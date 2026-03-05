"""
Download Etymological WordNet
=============================
Downloads the etymwn-20130208.zip from archive.org,
extracts etymwn.tsv into data/raw/.

Usage:
    python src/download_etymwn.py
"""

import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ETYMWN_URL, ETYMWN_FILE, RAW_DATA_DIR


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a simple progress indicator."""
    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  Downloading: {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)

    print(f"  URL: {url}")
    urlretrieve(url, str(dest), reporthook=_progress)
    print()  # newline after progress


def main():
    print("=" * 60)
    print("  Etymological WordNet Downloader")
    print("=" * 60)

    # Ensure raw data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if ETYMWN_FILE.exists():
        size_mb = ETYMWN_FILE.stat().st_size / (1024 * 1024)
        print(f"\n✓ etymwn.tsv already exists ({size_mb:.1f} MB)")
        print(f"  Location: {ETYMWN_FILE}")
        return

    # Download zip
    zip_path = RAW_DATA_DIR / "etymwn-20130208.zip"

    if not zip_path.exists():
        print(f"\n[1/2] Downloading etymwn-20130208.zip...")
        try:
            download_with_progress(ETYMWN_URL, zip_path)
            print(f"  ✓ Downloaded to {zip_path}")
        except Exception as e:
            print(f"\n  ✗ Download failed: {e}")
            print(f"\n  Manual download:")
            print(f"    1. Go to: {ETYMWN_URL}")
            print(f"    2. Save to: {zip_path}")
            sys.exit(1)
    else:
        print(f"\n[1/2] Zip already downloaded: {zip_path}")

    # Extract
    print(f"\n[2/2] Extracting etymwn.tsv...")
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            # Find the tsv file inside the zip
            tsv_files = [f for f in zf.namelist() if f.endswith('.tsv')]
            if not tsv_files:
                print("  ✗ No .tsv file found in zip!")
                sys.exit(1)

            tsv_name = tsv_files[0]
            print(f"  Found: {tsv_name}")

            # Extract to raw data dir
            zf.extract(tsv_name, str(RAW_DATA_DIR))

            # Rename if needed
            extracted_path = RAW_DATA_DIR / tsv_name
            if extracted_path != ETYMWN_FILE:
                extracted_path.rename(ETYMWN_FILE)

        size_mb = ETYMWN_FILE.stat().st_size / (1024 * 1024)
        print(f"  ✓ Extracted: {ETYMWN_FILE} ({size_mb:.1f} MB)")

    except zipfile.BadZipFile:
        print("  ✗ Corrupt zip file. Delete and re-download:")
        print(f"    del {zip_path}")
        sys.exit(1)

    # Clean up zip
    print(f"\n  Cleaning up zip file...")
    zip_path.unlink()
    print(f"  ✓ Removed {zip_path.name}")

    print(f"\n{'=' * 60}")
    print(f"  Done! etymwn.tsv is ready at:")
    print(f"  {ETYMWN_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
