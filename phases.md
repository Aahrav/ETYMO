PHASE 0 — Define Scope (DO THIS FIRST)

Why: Prevents the project from exploding.

Fixed scope decisions

Language: English only

Etymology classes:
{Germanic, Latin, Greek, Other}

Time periods:

Period A: 1800–1900 (Project Gutenberg)

Period B: 2000–2020 (Wikipedia / news)

Words tracked for semantic shift:
~100 words total (≈25 per origin)

👉 Write this clearly in your report.

🔹 PHASE 1 — Dataset Creation (Etymology)
1.1 Collect word–origin data

Sources:

Open etymology databases

Curated word lists:

“English words of Latin origin”

“English words of Greek origin”

“Germanic root words in English”

You only need 2–4k words total.

1.2 Normalize origin labels

Map fine-grained origins to coarse classes:

Original origin	Mapped class
Old English, Norse	Germanic
Latin, French	Latin
Ancient Greek	Greek
Arabic, Sanskrit, others	Other
1.3 Final dataset format (CSV)
word,origin_class
government,Latin
house,Germanic
biology,Greek
algebra,Other

Split:

80% train

20% test

🔹 PHASE 2 — Etymology Classifier (Core ML)
2.1 Feature extraction

Character-level n-grams

n = 2 to 5

TF-IDF weighting

Why?

Captures suffixes like -tion, -ology, -ship

Language-specific letter patterns

2.2 Model choice

Start with:

Logistic Regression (baseline)
Optional:

Linear SVM

2.3 Training & evaluation

Metrics:

Accuracy

Precision / Recall (per class)

Confusion Matrix

Expected accuracy:

70–85% (very good for this task)

2.4 Save trained model

You’ll reuse it later to label words.

🔹 PHASE 3 — Word Selection for Semantic Shift
3.1 Predict origins for candidate words

Take a large English word list

Run your classifier

Select high-confidence predictions (≥0.8 probability)

3.2 Manually filter (important)

Pick words that:

Appear in both corpora

Are not proper nouns

Have known semantic change potential

Final selection:

~25 Germanic

~25 Latin

~25 Greek

~25 Other

Save as:

word,origin_class
mouse,Germanic
stream,Germanic
culture,Latin
biology,Greek
🔹 PHASE 4 — Diachronic Corpus Preparation
4.1 Corpus A (Historical)

Project Gutenberg books

Focus: 1800–1900

Clean headers/footers

4.2 Corpus B (Modern)

Wikipedia dump OR news dataset

Clean markup

4.3 Preprocessing (both)

Lowercase

Tokenize

Remove punctuation

Sentence segmentation

⚠️ Keep preprocessing identical for both corpora.

🔹 PHASE 5 — Train Word Embeddings
5.1 Train two Word2Vec models

Skip-gram

vector size = 100

window = 5

min_count = 10

Models:

W2V_old

W2V_new

🔹 PHASE 6 — Embedding Alignment (KEY STEP)
6.1 Shared vocabulary extraction

Get intersection of words in both models

6.2 Anchor words

Use frequent, stable words:

the, and, of, man, water, sun, time

6.3 Orthogonal Procrustes alignment

Align W2V_old → W2V_new

Use scipy.linalg.orthogonal_procrustes

Explain in report:

“We rotate the historical embedding space to match the modern space while preserving distances.”

🔹 PHASE 7 — Semantic Drift Computation
7.1 Drift metric

For each word:

drift = 1 - cosine_similarity(vec_old, vec_new)
7.2 Neighbor comparison (qualitative)

For selected words:

Top-5 neighbors in old corpus

Top-5 neighbors in new corpus

Example:

mouse (1800s): trap, rat, vermin
mouse (2000s): click, cursor, device
🔹 PHASE 8 — Origin-Wise Analysis (THE NOVELTY)
8.1 Aggregate drift by origin

Compute:

Mean drift per origin

Variance

8.2 Compare groups

Answer:

Which origin drifts most?

Which is most stable?

This is your main result.

🔹 PHASE 9 — Visualization (UMAP + Manim)
9.1 Static plots (matplotlib/seaborn)

Drift bar chart: Origin vs average drift

Box plot: Drift distribution per origin class

9.2 UMAP dimensionality reduction

Use UMAP to reduce aligned embeddings to 2D

Plot both Period A and Period B embeddings in same 2D space

Color-code by origin class

Use anchor words as reference points (stable landmarks)

Why UMAP over PCA/t-SNE?

Preserves global structure better than t-SNE

Faster than t-SNE on larger vocabularies

Produces more interpretable 2D layouts

9.3 Manim animated visualizations

Animate semantic drift: words moving across the 2D UMAP plane
from Period A positions to Period B positions

Show arrows indicating direction and magnitude of drift

Clusters forming/dissolving between epochs

Color-code arrows by origin class to show origin-wise patterns

Incremental reveals: build up the visualization layer by layer

Recommended approach: Run UMAP first to get 2D coordinates,
then use Manim to animate the results — words drifting across
the plane between epochs, clusters forming, arrows showing
semantic movement.

🔹 PHASE 10 — Demo (Lightweight)

Choose one:

CLI:

Enter word → origin + drift score

Simple Flask app

Not required, but looks great.

🔹 PHASE 11 — Report Structure

Introduction

Problem statement

Related work (short)

System architecture

Etymology classification model

Diachronic embedding methodology

Semantic drift analysis

Results & discussion

Limitations

Future work

Conclusion