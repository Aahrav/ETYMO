# From Roots to Meaning: Etymology-Aware Semantic Shift Analysis

An NLP project that investigates how word meanings shift over time, and whether a word's etymological origin influences the degree of that shift.

## Project Scope

| Parameter | Value |
|---|---|
| **Language** | English only |
| **Etymology Classes** | Germanic · Latin · Greek · Other |
| **Historical Period** | 1800–1900 (Project Gutenberg) |
| **Modern Period** | 2000–2020 (Wikipedia / News) |
| **Words Tracked** | ~100 total (≈25 per origin class) |

## Methodology

1. **Etymology Classification** — Character n-gram features + TF-IDF → Logistic Regression
2. **Diachronic Embeddings** — Word2Vec Skip-gram trained on historical & modern corpora
3. **Embedding Alignment** — Orthogonal Procrustes to align vector spaces
4. **Semantic Drift** — `1 − cosine_similarity(vec_old, vec_new)` per word
5. **Origin-Wise Analysis** — Compare mean drift across etymology classes

## Directory Structure

```
ETYMO/
├── data/                # Etymology datasets (CSVs, word lists)
├── src/                 # Source code
│   ├── config.py        # Project-wide constants
│   ├── classifier.py    # Etymology classifier (Phase 2)
│   ├── preprocessing.py # Corpus preprocessing (Phase 4)
│   ├── embeddings.py    # Word2Vec training & alignment (Phases 5-6)
│   └── shift_analysis.py# Drift computation & analysis (Phases 7-8)
├── models/              # Saved model artifacts (.pkl, .model)
├── corpora/
│   ├── raw/             # Raw corpus text files
│   └── processed/       # Cleaned & tokenized corpus files
├── results/             # Drift scores, analysis outputs
├── notebooks/           # Jupyter notebooks for exploration
├── visualizations/      # Generated plots and Manim outputs
├── phases.md            # Full project plan (Phases 0–11)
├── SCOPE.md             # Formal scope definition
└── requirements.txt     # Python dependencies
```

## Quick Start

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## License

Academic project — all rights reserved.
