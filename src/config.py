"""
Project Configuration
=====================
Centralized constants and parameters for the Etymology-Aware
Semantic Shift Analysis project. All magic numbers live here.
"""

from pathlib import Path

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
