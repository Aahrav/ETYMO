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
ORIGIN_CLASSES = ["Germanic", "Latin", "Greek", "Sanskrit", "PIE", "Other"]

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
    # Sanskrit / Hindi
    "Sanskrit": "Sanskrit",
    "Hindi": "Sanskrit",
    "Pali": "Sanskrit",
    # PIE
    "Proto-Indo-European": "PIE",
    "PIE": "PIE",
    # Other
    "Arabic": "Other",
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
    # Sanskrit / Hindi
    "san": "Sanskrit",   # Sanskrit
    "hin": "Sanskrit",   # Hindi
    "pli": "Sanskrit",   # Pali
    # PIE
    "ine-pro": "PIE",    # Proto-Indo-European (ISO: ine-pro)
    # Other
    "ara": "Other",      # Arabic
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
TOTAL_SHIFT_WORDS = 150
WORDS_PER_ORIGIN = 25
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.8

# ──────────────────────────────────────────────
# Classifier Parameters (Phase 2)
# ──────────────────────────────────────────────
NGRAM_RANGE = (1, 6)          # Character n-gram range (widened from (2,5))
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
# Word Selection Parameters (Phase 3)
# ──────────────────────────────────────────────
SELECTED_WORDS_CSV = DATA_DIR / "selected_words.csv"
SELECTION_REPORT = RESULTS_DIR / "selection_report.txt"
MIN_WORD_LENGTH = 3          # Skip very short words (less reliable origins)
MAX_WORD_LENGTH = 20         # Skip very long/compound words

# Curated seed words with KNOWN interesting semantic histories.
# These are included regardless of classifier confidence to ensure
# we capture words with documented meaning changes.
SEED_WORDS = {
    "Germanic": [
        "mouse",    # vermin → computer device
        "web",      # spider web → internet
        "stream",   # water flow → data streaming
        "bug",      # insect → software error
        "cloud",    # sky → computing
        "ship",     # vessel → to send/deliver
        "broadcast",# scatter seed → radio/TV
        "board",    # plank → committee
        "craft",    # skill → vessel
        "barn",     # barley-house → farm building
    ],
    "Latin": [
        "virus",    # poison/slime → microbe → malware
        "computer", # one who computes → machine
        "cell",     # small room → biology → phone
        "text",     # woven thing → writing → SMS
        "monitor",  # one who warns → screen
        "server",   # one who serves → machine
        "tablet",   # writing slab → device
        "journal",  # daily → newspaper/diary
        "cabinet",  # small room → government body
        "culture",  # farming → arts/civilization
    ],
    "Greek": [
        "atom",     # indivisible → smallest particle
        "phone",    # voice → telephone → smartphone
        "organ",    # tool/instrument → body part → music
        "plasma",   # something molded → blood → TV type
        "energy",   # activity → physics concept
        "antenna",  # yard-arm → insect feeler → radio
        "icon",     # image → religious art → UI element
        "program",  # public notice → plan → software
        "cycle",    # circle → recurring sequence
        "meter",    # measure → unit → device
    ],
    "Sanskrit": [
        "avatar",   # descent of god → digital representation
        "guru",     # teacher → expert
        "karma",    # action → cause and effect
        "yoga",     # union → physical practice
        "nirvana",  # blowing out → mental state
        "jungle",   # jangal (Hindi) → wild forest
        "bandit",   # banjh (Hindi) → outlaw
        "shampoo",  # champo (Hindi) → hair wash
        "pajama",   # pa-jama (Persian/Hindi) → leg garment
        "punch",    # paunch (Sanskrit: five) → five-ingredient drink
    ],
    "PIE": [
        "father",   # *pater → ancestor
        "mother",   # *mater → maternal ancestor
        "water",    # *wod → liquid
        "new",      # *newo → fresh
        "three",    # *trei → numeral 3
        "night",    # *nokt → dark hours
        "eye",      # *okw → organ of sight
        "star",     # *ster → celestial body
        "heart",    # *kerd → core/organ
        "name",     # *nomn → identifier
    ],
    "Other": [
        "algebra",  # Arabic: reunion of broken parts → math
        "algorithm",# from 'al-Khwarizmi' → sequence of steps
        "magazine", # Arabic: storehouse → periodical
        "coffee",   # Arabic: qahwa → beverage
        "sugar",    # Sanskrit: sharkara → sweetener
        "candy",    # Arabic: qandi → sweets
        "jungle",   # Hindi: jangal → wild forest
        "avatar",   # Sanskrit: descent of god → digital representation
        "guru",     # Sanskrit: teacher → expert
        "tsunami",  # Japanese: harbor wave → disaster
    ],
}

# ──────────────────────────────────────────────
# Corpus Parameters (Phase 4)
# ──────────────────────────────────────────────
TARGET_TOKENS = 7_000_000        # Target ~7M tokens per corpus (middle of 5-10M range)
TOKEN_TOLERANCE = 0.15           # ±15% acceptable deviation
MIN_SENTENCE_TOKENS = 5          # Remove sentences with fewer tokens
MIN_TOKEN_LENGTH = 2             # Remove single-character tokens

# Gutenberg Historical Corpus (1800-1900)
GUTENBERG_BOOKS_TARGET = 150     # Download ~150 books
GUTENBERG_RAW = CORPORA_RAW_DIR / "gutenberg"
GUTENBERG_CATALOG_CACHE = CORPORA_RAW_DIR / "gutenberg_catalog.json"

# Wikipedia Modern Corpus (2000-2020)
WIKI_ARTICLES_TARGET = 50_000    # Process ~50K articles (more than enough)
WIKI_RAW = CORPORA_RAW_DIR / "wikipedia"

# Processed output paths
CORPUS_OLD = CORPORA_PROCESSED_DIR / "corpus_old.txt"
CORPUS_NEW = CORPORA_PROCESSED_DIR / "corpus_new.txt"
CORPUS_STATS = RESULTS_DIR / "corpus_stats.txt"

# ──────────────────────────────────────────────
# Word2Vec Parameters (Phase 5)
# ──────────────────────────────────────────────
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 10
W2V_SG = 1                    # 1 = Skip-gram, 0 = CBOW
W2V_EPOCHS = 15               # More than default 5 to compensate for smaller corpus
W2V_NEGATIVE = 10             # Negative sampling (higher = better for small corpora)
W2V_WORKERS = 4               # Parallel training threads

# Embedding output paths
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
W2V_MODEL_OLD = EMBEDDINGS_DIR / "w2v_old.model"
W2V_MODEL_NEW = EMBEDDINGS_DIR / "w2v_new.model"
W2V_ALIGNED_OLD = EMBEDDINGS_DIR / "w2v_old_aligned.npy"
EMBEDDING_REPORT = RESULTS_DIR / "embedding_report.txt"

# ──────────────────────────────────────────────
# Alignment (Phase 6)
# ──────────────────────────────────────────────
# Anchor words: frequent, semantically STABLE words assumed to
# NOT have changed meaning between 1800-1900 and 2000-2020.
# More anchors = better Procrustes alignment quality.
ANCHOR_WORDS = [
    # Function words (extremely stable)
    "the", "and", "of", "is", "in", "to", "for", "with", "that", "this",
    "was", "are", "but", "not", "all", "can", "had", "her", "one", "our",
    "will", "each", "make", "how", "like", "been", "has", "may",
    # Nature / physical world
    "man", "water", "sun", "time", "day", "hand", "house", "king", "land",
    "world", "god", "life", "earth", "fire", "stone", "tree", "river",
    "mountain", "sea", "wind", "rain", "sky", "star", "moon", "night",
    "morning", "summer", "winter", "snow", "field",
    # Body / human
    "head", "eye", "heart", "blood", "body", "face", "foot", "hair",
    "mouth", "name", "father", "mother", "brother", "sister", "son",
    "daughter", "child", "woman", "girl", "boy",
    # Abstract / common nouns
    "word", "thing", "place", "year", "work", "part", "way", "end",
    "number", "side", "mind", "power", "form", "point", "fact", "matter",
    "home", "room", "door", "table", "bed", "food", "bread", "horse",
    "dog", "book", "war", "death", "friend", "love", "truth", "law",
    "money", "gold", "silver", "iron", "wood", "road", "town", "church",
    # Common verbs
    "said", "come", "give", "take", "know", "think", "want", "look",
    "tell", "turn", "keep", "leave", "begin", "seem", "help", "talk",
    "stand", "hold", "move", "live", "believe", "bring", "happen",
    # Common adjectives
    "good", "great", "little", "old", "young", "long", "small", "large",
    "white", "black", "red", "dark", "high", "low", "full", "strong",
    "poor", "rich", "true", "free", "whole", "deep", "dead", "cold",
    "hard", "open", "close", "right", "wrong", "real", "sure",
]

# ──────────────────────────────────────────────
# Drift (Phase 7)
# ──────────────────────────────────────────────
TOP_N_NEIGHBORS = 5           # Nearest neighbors for qualitative comparison

