"""
Flask Web Application (Phase 10)
==================================
Interactive web dashboard for the Etymology-Aware Semantic Shift Analysis.

Serves:
  - Dashboard with key findings and embedded Manim animations
  - Word Explorer with origin prediction and drift analysis
  - Word Comparison with side-by-side analysis
  - Origin Overview with statistical results

Usage:
    python web/app.py
"""

import sys
import json
from pathlib import Path

from flask import Flask, render_template, jsonify, request, send_from_directory

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from src.config import (
    W2V_MODEL_OLD, W2V_MODEL_NEW, W2V_ALIGNED_OLD,
    RESULTS_DIR, ORIGIN_CLASSES, MODELS_DIR,
    SELECTED_WORDS_CSV,
)
from src.classifier import anchor

# ── Flask App ──
app = Flask(__name__,
    static_folder="static",
    template_folder="templates",
)

# ── Global data (loaded once at startup) ──
DATA = {}


def load_data():
    """Load all data at startup for fast responses."""
    print("  Loading data...")

    # Drift scores
    drift_df = pd.read_csv(RESULTS_DIR / "drift_scores.csv")
    DATA["drift_df"] = drift_df
    DATA["valid_drift"] = drift_df[drift_df["status"] == "OK"].copy()

    # Origin summary
    DATA["origin_summary"] = pd.read_csv(RESULTS_DIR / "origin_drift_summary.csv")

    # UMAP coordinates
    DATA["umap_df"] = pd.read_csv(RESULTS_DIR / "umap_coords.csv")

    # Selected words
    DATA["selected_words"] = pd.read_csv(SELECTED_WORDS_CSV)

    # Word2Vec models
    print("    Loading Word2Vec models...")
    DATA["model_old"] = Word2Vec.load(str(W2V_MODEL_OLD))
    DATA["model_new"] = Word2Vec.load(str(W2V_MODEL_NEW))
    DATA["aligned_old"] = np.load(str(W2V_ALIGNED_OLD))

    old_words = list(DATA["model_old"].wv.key_to_index.keys())
    DATA["old_words"] = old_words
    DATA["old_w2i"] = {w: i for i, w in enumerate(old_words)}

    # Classifier (if available)
    classifier_path = MODELS_DIR / "etymology_classifier.pkl"
    if classifier_path.exists():
        import joblib
        DATA["classifier"] = joblib.load(classifier_path)
        print("    ✓ Classifier loaded")
    else:
        DATA["classifier"] = None

    # Stats
    DATA["stats"] = {
        "total_words": len(DATA["valid_drift"]),
        "total_vocab_old": len(DATA["model_old"].wv),
        "total_vocab_new": len(DATA["model_new"].wv),
        "mean_drift": round(float(DATA["valid_drift"]["drift_score"].mean()), 4),
        "kruskal_p": 0.000,
    }

    # Neighbor analysis text
    nbr_path = RESULTS_DIR / "neighbor_analysis.txt"
    if nbr_path.exists():
        DATA["neighbor_text"] = nbr_path.read_text(encoding="utf-8")

    print(f"  ✓ Data loaded: {DATA['stats']['total_words']} tracked words")


def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def get_old_neighbors(word, topn=5):
    """Get neighbors from aligned old embedding space."""
    if word not in DATA["old_w2i"]:
        return []
    aligned = DATA["aligned_old"]
    target = aligned[DATA["old_w2i"][word]]
    norms = np.linalg.norm(aligned, axis=1)
    norms[norms == 0] = 1
    sims = aligned @ target / (norms * np.linalg.norm(target))
    top_idxs = np.argsort(sims)[::-1][1:topn + 1]
    return [(DATA["old_words"][i], round(float(sims[i]), 4)) for i in top_idxs]


def get_word_data(word):
    """Get comprehensive data for a single word."""
    word = word.lower().strip()
    drift_row = DATA["valid_drift"][DATA["valid_drift"]["word"] == word]
    result = {"word": word, "found": False}

    if len(drift_row) == 0:
        return result

    row = drift_row.iloc[0]
    result["found"] = True
    result["origin"] = row["origin_class"]
    result["drift_score"] = round(float(row["drift_score"]), 4)
    result["similarity"] = round(float(row["similarity"]), 4)
    result["confidence"] = round(float(row.get("confidence", 0)), 4)
    result["source"] = row.get("source", "unknown")

    # Classify drift level
    d = result["drift_score"]
    if d < 0.3:
        result["drift_level"] = "Stable"
        result["drift_color"] = "stable"
    elif d < 0.5:
        result["drift_level"] = "Low Shift"
        result["drift_color"] = "low"
    elif d < 0.7:
        result["drift_level"] = "Moderate Shift"
        result["drift_color"] = "moderate"
    else:
        result["drift_level"] = "High Shift"
        result["drift_color"] = "high"

    # Neighbors
    result["old_neighbors"] = get_old_neighbors(word)
    new_nbrs = DATA["model_new"].wv.most_similar(word, topn=5) \
        if word in DATA["model_new"].wv else []
    result["new_neighbors"] = [(w, round(s, 4)) for w, s in new_nbrs]

    # Neighbor overlap
    old_set = {w for w, _ in result["old_neighbors"]}
    new_set = {w for w, _ in result["new_neighbors"]}
    result["shared_neighbors"] = list(old_set & new_set)

    # Word-specific video
    # We use ROOT / "results" directly to ensure absolute path consistency
    word_video = ROOT / "results" / "videos" / "videos" / "words" / f"{word}.mp4"
    if word_video.exists():
        result["video_path"] = f"words/{word}.mp4"
    else:
        result["video_path"] = None

    return result


# ── Video path mapping ──
VIDEO_MAP = {
    "intro": "embeddings_intro_scene/720p30/EmbeddingsIntroScene.mp4",
    "drift": "drift_scene/720p30/DriftScene.mp4",
    "comparison": "comparison_scene/720p30/ComparisonScene.mp4",
    "origin_bars": "origin_bars_scene/720p30/OriginBarsScene.mp4",
    "alignment": "alignment_scene/720p30/AlignmentScene.mp4",
}


# ──────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    """Dashboard / Home page."""
    valid = DATA["valid_drift"].sort_values("drift_score", ascending=False)

    top_drifted = valid.head(5).to_dict("records")
    most_stable = valid.tail(5).to_dict("records")

    origin_data = DATA["origin_summary"].to_dict("records")

    return render_template("dashboard.html",
        stats=DATA["stats"],
        top_drifted=top_drifted,
        most_stable=most_stable,
        origin_data=origin_data,
        videos=VIDEO_MAP,
    )


@app.route("/explorer")
def explorer():
    """Word Explorer page."""
    word = request.args.get("word", "")
    word_data = get_word_data(word) if word else None

    all_words = sorted(DATA["valid_drift"]["word"].tolist())

    return render_template("explorer.html",
        word_data=word_data,
        all_words=all_words,
        query=word,
    )


@app.route("/compare")
def compare():
    """Word Comparison page."""
    word_a = request.args.get("a", "")
    word_b = request.args.get("b", "")

    data_a = get_word_data(word_a) if word_a else None
    data_b = get_word_data(word_b) if word_b else None

    # Cross-word similarity if both exist
    cross_sim = None
    if (data_a and data_a["found"] and data_b and data_b["found"]):
        w_a, w_b = word_a, word_b
        if w_a in DATA["old_w2i"] and w_b in DATA["old_w2i"]:
            va_old = DATA["aligned_old"][DATA["old_w2i"][w_a]]
            vb_old = DATA["aligned_old"][DATA["old_w2i"][w_b]]
            sim_old = cosine_sim(va_old, vb_old)
        else:
            sim_old = None

        if w_a in DATA["model_new"].wv and w_b in DATA["model_new"].wv:
            sim_new = cosine_sim(DATA["model_new"].wv[w_a], DATA["model_new"].wv[w_b])
        else:
            sim_new = None

        cross_sim = {
            "sim_old": round(sim_old, 4) if sim_old else None,
            "sim_new": round(sim_new, 4) if sim_new else None,
        }

    all_words = sorted(DATA["valid_drift"]["word"].tolist())

    return render_template("compare.html",
        data_a=data_a,
        data_b=data_b,
        cross_sim=cross_sim,
        all_words=all_words,
        query_a=word_a,
        query_b=word_b,
        videos=VIDEO_MAP,
    )


@app.route("/overview")
def overview():
    """Origin Overview page."""
    origin_data = DATA["origin_summary"].to_dict("records")

    # Per-origin top words
    per_origin = {}
    valid = DATA["valid_drift"]
    for origin in ORIGIN_CLASSES:
        og = valid[valid["origin_class"] == origin].sort_values(
            "drift_score", ascending=False
        )
        per_origin[origin] = og.head(5).to_dict("records")

    # UMAP data for scatter
    umap_data = DATA["umap_df"].to_dict("records")

    return render_template("overview.html",
        origin_data=origin_data,
        per_origin=per_origin,
        umap_data=json.dumps(umap_data),
        stats=DATA["stats"],
        videos=VIDEO_MAP,
    )


@app.route("/predict")
def predict():
    """Etymology Predictor page."""
    word = request.args.get("word", "").strip()
    result = None

    if word and DATA["classifier"] is not None:
        anchored = anchor(word)
        proba = DATA["classifier"].predict_proba([anchored])[0]
        classes = DATA["classifier"].classes_
        prob_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}
        predicted = classes[proba.argmax()]

        # Sort probabilities descending for the bar chart
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        # Extract sample n-grams for educational display
        sample_ngrams = []
        anch = f"^{word}$"
        for n in range(2, 6):
            for i in range(len(anch) - n + 1):
                ng = anch[i:i + n]
                if len(sample_ngrams) < 12:
                    sample_ngrams.append(ng)

        # Vocabulary size from the vectorizer
        try:
            vocab_size = len(DATA["classifier"].named_steps["tfidf"].vocabulary_)
        except Exception:
            vocab_size = "18,000+"

        result = {
            "word": word,
            "predicted": predicted,
            "probabilities": prob_dict,
            "sorted_probs": sorted_probs,
            "sample_ngrams": sample_ngrams,
        }

    # Example words for the landing state
    examples = [
        ("algorithm", "Other", "Arabic al-Khwārizmī → math"),
        ("shampoo", "Sanskrit", "Hindi chāmpo → hair wash"),
        ("father", "PIE", "*pater → ancestor"),
        ("telephone", "Greek", "tele + phone → far voice"),
        ("virus", "Latin", "poison → microbe → malware"),
        ("stream", "Germanic", "water flow → data streaming"),
    ]

    # Suggestion words for the result state
    suggestions = ["democracy", "jungle", "algebra", "yoga",
                    "mother", "craft", "tsunami", "monitor",
                    "nirvana", "culture", "atom", "magazine"]

    # Vocab size for display
    try:
        vocab_size = len(DATA["classifier"].named_steps["tfidf"].vocabulary_)
    except Exception:
        vocab_size = "18,000+"

    return render_template("predict.html",
        result=result,
        query=word,
        examples=examples,
        suggestions=suggestions,
        stats={"vocab_size": vocab_size},
    )


# ── API routes for AJAX ──
@app.route("/api/word/<word>")
def api_word(word):
    """API endpoint for word data."""
    return jsonify(get_word_data(word))


@app.route("/api/search")
def api_search():
    """Autocomplete search."""
    q = request.args.get("q", "").lower()
    if len(q) < 1:
        return jsonify([])
    matches = [w for w in DATA["valid_drift"]["word"].tolist() if w.startswith(q)]
    return jsonify(matches[:10])


@app.route("/api/classify/<word>")
def api_classify(word):
    """Classify a word's etymology."""
    if DATA["classifier"] is None:
        return jsonify({"error": "Classifier not loaded"})
    anchored = anchor(word)
    proba = DATA["classifier"].predict_proba([anchored])[0]
    classes = DATA["classifier"].classes_
    result = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    predicted = classes[proba.argmax()]
    return jsonify({"word": word, "predicted": predicted, "probabilities": result})


# ── Serve Manim videos from results directory ──
@app.route("/videos/<path:filename>")
def serve_video(filename):
    """Serve Manim video files."""
    # Manim nests a 'videos' dir inside its media_dir
    return send_from_directory(str(RESULTS_DIR / "videos" / "videos"), filename)


# ── Serve static figures ──
@app.route("/figures/<path:filename>")
def serve_figure(filename):
    """Serve static plot images."""
    return send_from_directory(str(RESULTS_DIR / "figures"), filename)


if __name__ == "__main__":
    load_data()
    print("\n  ✓ Starting ETYMO web app on http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)
