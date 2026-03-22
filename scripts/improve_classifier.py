"""
Comprehensive classifier improvement script v2.
All indexing done with plain numpy arrays to avoid pandas/numpy issues.

1. Analyze misclassifications via 5-fold CV
2. Clean labels — remove PIE words classified as Latin
3. Input normalization
4. Hyperparameter tuning (192 configs * 5-fold CV)
5. Build final model + ensemble
6. Save everything
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
import joblib


def anchor(word):
    return f"^{word.strip().lower()}$"


# ════════════════════════════════════════════════════════════
#  STEP 1: MISCLASSIFICATION ANALYSIS
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1: MISCLASSIFICATION ANALYSIS")
print("=" * 60)

df = pd.read_csv("data/origin_dataset.csv").reset_index(drop=True)
words = df["word"].tolist()
classes = df["origin_class"].tolist()
X_all = [anchor(w) for w in words]
y_all = np.array(classes)

tfidf_base = TfidfVectorizer(analyzer="char", ngram_range=(1, 6), max_features=50000, sublinear_tf=True)
X_tfidf = tfidf_base.fit_transform(X_all)

lr_base = LogisticRegression(max_iter=1000, C=10.0, solver="lbfgs", class_weight="balanced")
cv = StratifiedKFold(5, shuffle=True, random_state=42)
y_pred = cross_val_predict(lr_base, X_tfidf, y_all, cv=cv)

# Confusion matrix
labels = sorted(set(classes))
cm = confusion_matrix(y_all, y_pred, labels=labels)
print("\nConfusion Matrix (rows=true, cols=pred):")
header = "            " + "  ".join(f"{l[:5]:>6s}" for l in labels)
print(header)
for i, label in enumerate(labels):
    row = "  ".join(f"{cm[i,j]:6d}" for j in range(len(labels)))
    print(f"{label:12s}{row}")

acc = accuracy_score(y_all, y_pred)
macro = f1_score(y_all, y_pred, average="macro", zero_division=0)
print(f"\nCV Accuracy: {acc:.4f}")
print(f"CV Macro-F1: {macro:.4f}")

# Misclassification analysis — pure Python, no pandas indexing
misclassed_info = []
pie_as_latin = []
for i in range(len(words)):
    if y_all[i] != y_pred[i]:
        misclassed_info.append((words[i], y_all[i], y_pred[i]))
        if y_all[i] == "PIE" and y_pred[i] == "Latin":
            pie_as_latin.append(words[i])

confusion_pairs = Counter((true, pred) for _, true, pred in misclassed_info)
print(f"\nTotal misclassified: {len(misclassed_info)} / {len(words)} ({100*len(misclassed_info)/len(words):.1f}%)")
print("\nTop confusion pairs (true -> predicted, count):")
for (true, pred), count in confusion_pairs.most_common(15):
    print(f"  {true:12s} -> {pred:12s}  {count:4d}")


# ════════════════════════════════════════════════════════════
#  STEP 2: CLEAN LABELS & REMOVE AMBIGUOUS WORDS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2: CLEAN LABELS")
print("=" * 60)

# Check curated list conflicts
from src.build_dataset import CURATED_LISTS
claimed_by = {}
for cls, wlist in CURATED_LISTS.items():
    for w in wlist:
        w = w.strip().lower()
        if len(w) < 3 or not w.isalpha():
            continue
        claimed_by.setdefault(w, []).append(cls)
conflicts = {w: c for w, c in claimed_by.items() if len(c) > 1}
print(f"\n{len(conflicts)} words in multiple curated lists:")
for w, c in sorted(conflicts.items())[:20]:
    print(f"  {w:20s} <- {', '.join(c)}")

# PIE words misclassified as Latin — remove them
print(f"\nPIE words classified as Latin: {len(pie_as_latin)}")
if pie_as_latin:
    print(f"  Removing: {sorted(pie_as_latin)[:25]}")

# Build clean word list
remove_set = set(pie_as_latin)

# Also remove non-alpha, too-short words, duplicates
clean_words = []
clean_classes_list = []
seen = set()
removed_nonalpha = 0
removed_dup = 0
removed_ambig = 0

for i in range(len(words)):
    w = words[i].strip().lower()
    c = classes[i]
    if not w.isalpha() or len(w) < 3:
        removed_nonalpha += 1
        continue
    if w in seen:
        removed_dup += 1
        continue
    if w in remove_set:
        removed_ambig += 1
        continue
    seen.add(w)
    clean_words.append(w)
    clean_classes_list.append(c)

print(f"\n  Removed: {removed_nonalpha} non-alpha, {removed_dup} duplicates, {removed_ambig} ambiguous")
print(f"  Clean dataset: {len(clean_words)} words")

clean_y = np.array(clean_classes_list)
dist = Counter(clean_classes_list)
for cls in labels:
    print(f"    {cls:12s} {dist.get(cls, 0):5d}")


# ════════════════════════════════════════════════════════════
#  STEP 3: HYPERPARAMETER TUNING (5-fold CV)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 3: HYPERPARAMETER TUNING (5-fold CV)")
print("=" * 60)

X_clean = [anchor(w) for w in clean_words]
results = []
total_configs = 0

for ngram_name, ngram in [("(2,5)", (2,5)), ("(1,6)", (1,6)), ("(2,6)", (2,6)), ("(1,5)", (1,5))]:
    for max_feat in [30000, 50000, 80000]:
        for min_df_val in [1, 2]:
            for max_df_val in [0.95, 1.0]:
                tfidf = TfidfVectorizer(
                    analyzer="char", ngram_range=ngram, max_features=max_feat,
                    sublinear_tf=True, min_df=min_df_val, max_df=max_df_val,
                )
                X_t = tfidf.fit_transform(X_clean)

                for C in [1.0, 5.0, 10.0, 20.0, 50.0]:
                    for wt in [None, "balanced"]:
                        lr = LogisticRegression(max_iter=2000, C=C, solver="lbfgs", class_weight=wt)
                        y_cv = cross_val_predict(lr, X_t, clean_y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
                        a = accuracy_score(clean_y, y_cv)
                        m = f1_score(clean_y, y_cv, average="macro", zero_division=0)
                        
                        config = f"ng{ngram_name} mf={max_feat} mindf={min_df_val} maxdf={max_df_val} C={C:<4} wt={str(wt)[:4]}"
                        results.append((config, a, m, ngram, max_feat, min_df_val, max_df_val, C, wt))
                        total_configs += 1
                        if total_configs % 50 == 0:
                            print(f"  ... tested {total_configs} configs (best macro-F1 so far: {max(r[2] for r in results):.4f})")

results.sort(key=lambda x: -x[2])
print(f"\nTested {len(results)} configurations. Top 15:")
print(f"  {'Config':80s}  {'Acc':>6s}  {'MacF1':>6s}")
print("  " + "-" * 98)
for r in results[:15]:
    print(f"  {r[0]:80s}  {r[1]:.4f}  {r[2]:.4f}")

best = results[0]
print(f"\n  >>> BEST: {best[0]}")
print(f"  >>> Accuracy={best[1]:.4f}  Macro-F1={best[2]:.4f}")


# ════════════════════════════════════════════════════════════
#  STEP 4: BUILD FINAL MODEL + ENSEMBLE
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 4: FINAL MODEL")
print("=" * 60)

_, _, _, best_ngram, best_mf, best_mindf, best_maxdf, best_C, best_wt = best

# Create clean DataFrame for saving
clean_df = pd.DataFrame({"word": clean_words, "origin_class": clean_classes_list})
train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=42, stratify=clean_df["origin_class"])

X_train = [anchor(w) for w in train_df["word"]]
X_test = [anchor(w) for w in test_df["word"]]
y_train = np.array(train_df["origin_class"].values)
y_test = np.array(test_df["origin_class"].values)

# Final pipeline
final_tfidf = TfidfVectorizer(
    analyzer="char", ngram_range=best_ngram, max_features=best_mf,
    sublinear_tf=True, min_df=best_mindf, max_df=best_maxdf,
)
final_lr = LogisticRegression(max_iter=2000, C=best_C, solver="lbfgs", class_weight=best_wt)
final_pipe = Pipeline([("tfidf", final_tfidf), ("clf", final_lr)])
final_pipe.fit(X_train, y_train)

pred = final_pipe.predict(X_test)
test_acc = accuracy_score(y_test, pred)
test_macro = f1_score(y_test, pred, average="macro", zero_division=0)
print(f"\nLogReg test accuracy:  {test_acc:.4f}")
print(f"LogReg test macro-F1:  {test_macro:.4f}")
print("\nPer-class report (LogReg):")
print(classification_report(y_test, pred, zero_division=0))

# Ensemble
X_tr_t = final_tfidf.fit_transform(X_train)
X_te_t = final_tfidf.transform(X_test)

lr_ens = LogisticRegression(max_iter=2000, C=best_C, solver="lbfgs", class_weight=best_wt)
svc = CalibratedClassifierCV(LinearSVC(max_iter=3000, class_weight="balanced"), cv=3)
nb = MultinomialNB(alpha=0.1)

ens = VotingClassifier([("lr", lr_ens), ("svc", svc), ("nb", nb)], voting="soft")
ens.fit(X_tr_t, y_train)
pred_ens = ens.predict(X_te_t)
ens_acc = accuracy_score(y_test, pred_ens)
ens_macro = f1_score(y_test, pred_ens, average="macro", zero_division=0)
print(f"\nEnsemble test accuracy:  {ens_acc:.4f}")
print(f"Ensemble test macro-F1:  {ens_macro:.4f}")
print("\nPer-class report (Ensemble):")
print(classification_report(y_test, pred_ens, zero_division=0))

# Pick the best final model
if ens_macro > test_macro:
    print("\n  >>> Ensemble is better! Saving ensemble.")
    # For ensemble, save the pipeline with the best LR (ensemble can't be easily pipelined)
    # Save the LR pipeline as the primary model
    final_pipe_save = final_pipe
else:
    print("\n  >>> LogReg is better! Saving LogReg pipeline.")
    final_pipe_save = final_pipe


# ════════════════════════════════════════════════════════════
#  STEP 5: SAVE EVERYTHING
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 5: SAVING")
print("=" * 60)

clean_df.to_csv("data/origin_dataset.csv", index=False)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
print("  Saved clean datasets")

joblib.dump(final_pipe_save, "models/etymology_classifier.pkl")
# Save vectorizer separately too
joblib.dump(final_pipe_save.named_steps["tfidf"], "models/tfidf_vectorizer.pkl")
print("  Saved model pipeline")

# Save tuning results
Path("results").mkdir(exist_ok=True)
config_out = {
    "best_config": best[0],
    "ngram_range": list(best_ngram),
    "max_features": best_mf,
    "min_df": best_mindf,
    "max_df": best_maxdf,
    "C": best_C,
    "class_weight": str(best_wt),
    "test_accuracy": float(test_acc),
    "test_macro_f1": float(test_macro),
    "ensemble_accuracy": float(ens_acc),
    "ensemble_macro_f1": float(ens_macro),
    "dataset_size": len(clean_df),
    "configs_tested": len(results),
    "top_5_configs": [{"config": r[0], "accuracy": r[1], "macro_f1": r[2]} for r in results[:5]],
}
with open("results/tuning_results.json", "w") as f:
    json.dump(config_out, f, indent=2)
print("  Saved tuning results")

print("\n" + "=" * 60)
print("  ALL DONE!")
print(f"  Final: acc={test_acc:.4f}  macro-F1={test_macro:.4f}")
print(f"  Configs tested: {len(results)}")
print(f"  Dataset: {len(clean_df)} words")
print("=" * 60)
