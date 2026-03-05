# Project Scope Definition

> This document formally defines the scope boundaries for the
> **Etymology-Aware Semantic Shift Analysis** project.
> All decisions below are fixed to prevent scope creep.

---

## 1. Language

**English only.**

No multilingual analysis. All corpora, word lists, and models target English.

---

## 2. Etymology Classes

Four coarse-grained origin classes:

| Class | Includes |
|---|---|
| **Germanic** | Old English, Old Norse, Dutch, German roots |
| **Latin** | Latin, Old French, Anglo-Norman, Romance roots |
| **Greek** | Ancient Greek, Hellenistic Greek roots |
| **Other** | Arabic, Sanskrit, Hindi, Japanese, and all remaining origins |

### Origin Mapping Rules

| Fine-Grained Origin | → Mapped Class |
|---|---|
| Old English, Norse, Dutch | Germanic |
| Latin, French, Norman | Latin |
| Ancient Greek | Greek |
| Arabic, Sanskrit, others | Other |

---

## 3. Time Periods

Two diachronic periods for comparison:

| Period | Years | Source |
|---|---|---|
| **Period A** (Historical) | 1800–1900 | Project Gutenberg |
| **Period B** (Modern) | 2000–2020 | Wikipedia / News corpora |

---

## 4. Word Selection for Semantic Shift

- **Total words tracked:** ~100
- **Per origin class:** ≈25 words
- **Selection criteria:**
  - High-confidence classifier prediction (≥0.8 probability)
  - Appears in both Period A and Period B corpora
  - Not a proper noun
  - Known semantic change potential

---

## 5. Model & Embedding Parameters

| Parameter | Value |
|---|---|
| **Classifier** | Logistic Regression |
| **Features** | Character n-grams (n=2–5), TF-IDF weighted |
| **Dataset size** | 2,000–4,000 words |
| **Train/Test split** | 80% / 20% |
| **Embedding model** | Word2Vec (Skip-gram) |
| **Vector dimensions** | 100 |
| **Context window** | 5 |
| **Minimum word count** | 10 |
| **Alignment method** | Orthogonal Procrustes |

---

## 6. Drift Metric

```
drift(w) = 1 − cosine_similarity(vec_old(w), vec_new(w))
```

Where `vec_old` and `vec_new` are aligned embeddings from Period A and Period B respectively.

---

## 7. Deliverables

1. Etymology classification model with evaluation metrics
2. Aligned diachronic word embeddings (2 models)
3. Semantic drift scores for ~100 words
4. Origin-wise drift comparison (mean, variance per class)
5. Visualizations (bar charts, PCA/t-SNE plots)
6. Final report following the structure in `phases.md` (Phase 11)
