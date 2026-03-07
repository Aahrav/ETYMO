"""
Project Configuration
=====================
Centralized constants and parameters for the Etymology-Aware
Semantic Shift Analysis project. All magic numbers live here.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CORPORA_RAW_DIR = PROJECT_ROOT / "corpora" / "raw"
CORPORA_PROCESSED_DIR = PROJECT_ROOT / "corpora" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"

# ──────────────────────────────────────────────
# Scope — Etymology Classes
# ──────────────────────────────────────────────
ORIGIN_CLASSES = ["Germanic", "Latin", "Greek", "Other"]

# Fine-grained → coarse mapping
ORIGIN_MAPPING = {
    # Germanic
    "Old English": "Germanic",
    "Middle English": "Germanic",
    "Old Norse": "Germanic",
    "Dutch": "Germanic",
    "German": "Germanic",
    "Norse": "Germanic",
    # Latin
    "Latin": "Latin",
    "French": "Latin",
    "Old French": "Latin",
    "Anglo-Norman": "Latin",
    "Norman": "Latin",
    "Italian": "Latin",
    "Spanish": "Latin",
    "Portuguese": "Latin",
    # Greek
    "Ancient Greek": "Greek",
    "Greek": "Greek",
    # Other
    "Arabic": "Other",
    "Sanskrit": "Other",
    "Hindi": "Other",
    "Japanese": "Other",
    "Persian": "Other",
    "Turkish": "Other",
    "Chinese": "Other",
}

# ISO 639-3 language codes → coarse origin class
# (Used to parse etymwn.tsv which stores 3-letter codes)
ISO_TO_ORIGIN = {
    # Germanic
    "ang": "Germanic",   # Old English
    "enm": "Germanic",   # Middle English
    "non": "Germanic",   # Old Norse
    "nld": "Germanic",   # Dutch
    "deu": "Germanic",   # German
    "gml": "Germanic",   # Middle Low German
    "gmh": "Germanic",   # Middle High German
    "odt": "Germanic",   # Old Dutch
    "goh": "Germanic",   # Old High German
    "osx": "Germanic",   # Old Saxon
    "frk": "Germanic",   # Frankish
    "sco": "Germanic",   # Scots
    "isl": "Germanic",   # Icelandic
    "swe": "Germanic",   # Swedish
    "dan": "Germanic",   # Danish
    "nor": "Germanic",   # Norwegian
    # Latin
    "lat": "Latin",      # Latin
    "fra": "Latin",      # French
    "fro": "Latin",      # Old French
    "xno": "Latin",      # Anglo-Norman
    "ita": "Latin",      # Italian
    "spa": "Latin",      # Spanish
    "por": "Latin",      # Portuguese
    "ron": "Latin",      # Romanian
    "cat": "Latin",      # Catalan
    "oci": "Latin",      # Occitan
    "pro": "Latin",      # Old Provençal
    "roa": "Latin",      # Romance (generic)
    "frm": "Latin",      # Middle French
    "nrf": "Latin",      # Norman French
    "glg": "Latin",      # Galician
    # Greek
    "grc": "Greek",      # Ancient Greek
    "ell": "Greek",      # Modern Greek
    # Other
    "ara": "Other",      # Arabic
    "san": "Other",      # Sanskrit
    "hin": "Other",      # Hindi
    "jpn": "Other",      # Japanese
    "fas": "Other",      # Persian
    "tur": "Other",      # Turkish
    "zho": "Other",      # Chinese
    "rus": "Other",      # Russian
    "heb": "Other",      # Hebrew
    "kor": "Other",      # Korean
    "mal": "Other",      # Malay
    "tgl": "Other",      # Tagalog
    "swa": "Other",      # Swahili
    "urd": "Other",      # Urdu
    "tam": "Other",      # Tamil
    "nah": "Other",      # Nahuatl
    "que": "Other",      # Quechua
    "yid": "Other",      # Yiddish
}

# ──────────────────────────────────────────────
# Dataset Paths
# ──────────────────────────────────────────────
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ETYMWN_URL = "https://archive.org/download/etymwn-20130208/etymwn-20130208.zip"
ETYMWN_FILE = RAW_DATA_DIR / "etymwn.tsv"
ORIGIN_DATASET = DATA_DIR / "origin_dataset.csv"
TRAIN_DATASET = DATA_DIR / "train.csv"
TEST_DATASET = DATA_DIR / "test.csv"
DATASET_STATS = DATA_DIR / "dataset_stats.txt"

# ──────────────────────────────────────────────
# Scope — Time Periods
# ──────────────────────────────────────────────
PERIOD_A = {"name": "Historical", "start": 1800, "end": 1900, "source": "Project Gutenberg"}
PERIOD_B = {"name": "Modern", "start": 2000, "end": 2020, "source": "Wikipedia / News"}

# ──────────────────────────────────────────────
# Scope — Word Selection
# ──────────────────────────────────────────────
TOTAL_SHIFT_WORDS = 100
WORDS_PER_ORIGIN = 25
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.8

# ──────────────────────────────────────────────
# Classifier Parameters (Phase 2)
# ──────────────────────────────────────────────
NGRAM_RANGE = (2, 5)          # Character n-gram range
TRAIN_TEST_SPLIT = 0.8        # 80% train, 20% test
CLASSIFIER_TYPE = "LogisticRegression"
CV_FOLDS = 5                  # Stratified K-Fold cross-validation

# Classifier output paths
CLASSIFIER_MODEL = MODELS_DIR / "etymology_classifier.pkl"
TFIDF_VECTORIZER = MODELS_DIR / "tfidf_vectorizer.pkl"
CLASSIFIER_REPORT = RESULTS_DIR / "classifier_report.txt"
CONFUSION_MATRIX_PNG = RESULTS_DIR / "confusion_matrix.png"
MISCLASSIFIED_CSV = RESULTS_DIR / "misclassified_words.csv"
MODEL_COMPARISON_JSON = RESULTS_DIR / "model_comparison.json"

# ──────────────────────────────────────────────
# Word2Vec Parameters (Phase 5)
# ──────────────────────────────────────────────
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 10
W2V_SG = 1                    # 1 = Skip-gram, 0 = CBOW

# ──────────────────────────────────────────────
# Alignment (Phase 6)
# ──────────────────────────────────────────────
ANCHOR_WORDS = [
    "the", "and", "of", "is", "in", "to", "a",
    "man", "water", "sun", "time", "day", "hand",
    "house", "king", "land", "world", "god", "life",
]

# ──────────────────────────────────────────────
# Drift (Phase 7)
# ──────────────────────────────────────────────
TOP_N_NEIGHBORS = 5           # Nearest neighbors for qualitative comparison
