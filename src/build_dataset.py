"""
Build Etymology Dataset (Phase 1)
==================================
End-to-end pipeline that:
  1. Parses etymwn.tsv to extract English word origins
  2. Normalizes origins to {Germanic, Latin, Greek, Other}
  3. Deduplicates and resolves conflicts
  4. Supplements underrepresented classes with curated word lists
  5. Balances classes and creates final dataset
  6. Splits into train/test (stratified 80/20)
  7. Generates dataset statistics

Usage:
    python src/build_dataset.py

Prerequisites:
    Run `python src/download_etymwn.py` first to get etymwn.tsv
"""

import re
import sys
import random
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    ISO_TO_ORIGIN, ORIGIN_CLASSES, ETYMWN_FILE,
    ORIGIN_DATASET, TRAIN_DATASET, TEST_DATASET, DATASET_STATS,
    DATA_DIR, TRAIN_TEST_SPLIT,
)

# ──────────────────────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Target dataset size per class
MIN_PER_CLASS = 400
MAX_PER_CLASS = 1200
TARGET_PER_CLASS = 800


# ──────────────────────────────────────────────────────────────
#  Step 1: Parse etymwn.tsv
# ──────────────────────────────────────────────────────────────
def parse_etymwn(filepath: Path) -> dict[str, list[str]]:
    """
    Parse the Etymological WordNet TSV file.

    Each line has format:
        eng:word    rel:etymology    lang:ancestor_word
        eng:word    rel:has_derived_form    lang:derived_word
        eng:word    rel:etymologically_related    lang:related_word

    We extract rows where:
      - Source is English (eng:)
      - Relationship is rel:etymology (word derives FROM another language)
      - Target language is in our ISO_TO_ORIGIN mapping

    Returns:
        dict mapping english_word -> list of origin language codes
    """
    print("\n[Step 1] Parsing etymwn.tsv...")

    if not filepath.exists():
        print(f"  ✗ File not found: {filepath}")
        print(f"  Run `python src/download_etymwn.py` first!")
        sys.exit(1)

    word_origins: dict[str, list[str]] = defaultdict(list)
    total_lines = 0
    matched_lines = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total_lines += 1
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue

            source, relation, target = parts

            # We want: English word that etymologically derives FROM another language
            # rel:etymology means "source derives from target"
            if not source.startswith("eng:"):
                continue
            if relation != "rel:etymology":
                continue

            # Extract the target language code
            if ":" not in target:
                continue
            target_lang = target.split(":")[0]

            # Skip self-references (English from English)
            if target_lang == "eng":
                continue

            # Only keep languages we know how to map
            if target_lang in ISO_TO_ORIGIN:
                word = source[4:]  # strip "eng:"
                word_origins[word].append(target_lang)
                matched_lines += 1

    print(f"  Total lines read:   {total_lines:,}")
    print(f"  Matched entries:    {matched_lines:,}")
    print(f"  Unique English words: {len(word_origins):,}")

    return dict(word_origins)


# ──────────────────────────────────────────────────────────────
#  Step 2: Normalize origins
# ──────────────────────────────────────────────────────────────
def normalize_origins(word_origins: dict[str, list[str]]) -> dict[str, str]:
    """
    Map each word to a single coarse origin class.

    Resolution strategy for words with multiple etymologies:
      - Map all ISO codes to coarse classes
      - Take majority vote
      - Break ties by priority: Latin > Germanic > Greek > Other
        (reflects that Latin/French is the most common indirect ancestor)
    """
    print("\n[Step 2] Normalizing origins...")

    priority = {"PIE": 0, "Sanskrit": 1, "Latin": 2, "Germanic": 3, "Greek": 4, "Other": 5}
    word_to_class: dict[str, str] = {}
    conflicts = 0

    for word, lang_codes in word_origins.items():
        # Map all codes to coarse classes
        classes = [ISO_TO_ORIGIN[code] for code in lang_codes]

        if len(set(classes)) == 1:
            # Unanimous
            word_to_class[word] = classes[0]
        else:
            # Conflict — majority vote, tie-break by priority
            conflicts += 1
            counts = Counter(classes)
            max_count = max(counts.values())
            candidates = [c for c, n in counts.items() if n == max_count]
            # Sort by priority, pick first
            candidates.sort(key=lambda c: priority.get(c, 99))
            word_to_class[word] = candidates[0]

    print(f"  Words with single origin:   {len(word_to_class) - conflicts:,}")
    print(f"  Words with conflicts resolved: {conflicts:,}")

    return word_to_class


# ──────────────────────────────────────────────────────────────
#  Step 3: Clean & filter
# ──────────────────────────────────────────────────────────────
def clean_words(word_to_class: dict[str, str]) -> dict[str, str]:
    """
    Remove words that are:
      - Not purely alphabetic (no numbers, hyphens, etc.)
      - Too short (< 3 chars)
      - Too long (> 25 chars)
      - Likely proper nouns (starts with uppercase in source data)
      - Common stop words
    """
    print("\n[Step 3] Cleaning and filtering words...")

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "shall", "must",
        "not", "no", "nor", "and", "or", "but", "if", "then",
        "than", "too", "very", "just", "about", "above", "after",
    }

    before = len(word_to_class)
    cleaned = {}

    for word, origin in word_to_class.items():
        w = word.strip().lower()

        # Must be purely alphabetic
        if not w.isalpha():
            continue
        # Length bounds
        if len(w) < 3 or len(w) > 25:
            continue
        # Skip stop words
        if w in stop_words:
            continue
        # Skip if already seen (dedup after lowercasing)
        if w in cleaned:
            continue

        cleaned[w] = origin

    after = len(cleaned)
    print(f"  Before cleaning: {before:,}")
    print(f"  After cleaning:  {after:,}")
    print(f"  Removed:         {before - after:,}")

    return cleaned


# ──────────────────────────────────────────────────────────────
#  Step 4: Curated supplemental word lists
# ──────────────────────────────────────────────────────────────
# These fill gaps in etymwn.tsv, especially for Greek and Other

CURATED_GERMANIC = [
    "book", "bread", "bridge", "child", "church", "cold", "dale", "dark",
    "deal", "death", "deep", "deer", "dirt", "door", "dream", "drink",
    "drive", "earth", "east", "eat", "edge", "egg", "end", "eye",
    "fall", "farm", "fast", "father", "fear", "field", "fight", "fire",
    "fish", "flat", "flesh", "flood", "floor", "fly", "fold", "folk",
    "food", "foot", "forget", "fox", "free", "freeze", "friend", "frost",
    "gain", "gate", "gift", "girl", "glad", "glass", "glove", "goat",
    "gold", "good", "grass", "grave", "green", "grip", "ground", "grow",
    "guest", "guilt", "gun", "hair", "half", "hall", "hammer", "hand",
    "hang", "happen", "happy", "hard", "harm", "harvest", "hate", "hawk",
    "head", "heal", "health", "heap", "hear", "heart", "heat", "heaven",
    "heavy", "heel", "help", "hen", "hide", "high", "hill", "hold",
    "hole", "home", "honey", "hood", "hook", "hope", "horn", "horse",
    "hound", "house", "hunger", "hunt", "husband", "ice", "iron",
    "island", "keen", "keep", "kernel", "kettle", "kind", "king",
    "kiss", "knee", "knife", "knot", "know", "lake", "lamb", "land",
    "last", "late", "law", "lead", "leaf", "learn", "leather", "leave",
    "length", "life", "lift", "light", "like", "limb", "little", "live",
    "load", "loaf", "lock", "long", "lord", "lose", "love", "low",
    "luck", "lung", "make", "man", "mare", "mark", "meal", "meet",
    "melt", "middle", "might", "mild", "milk", "mill", "mind", "moon",
    "moor", "morning", "mother", "mouth", "narrow", "near", "neck",
    "needle", "nest", "net", "new", "night", "north", "nose", "oath",
    "one", "open", "oven", "owl", "own", "path", "plough", "quick",
    "rain", "read", "red", "ride", "right", "ring", "rise", "road",
    "roof", "root", "rope", "rough", "round", "row", "run", "rust",
    "saddle", "sail", "salt", "sand", "sea", "seed", "seek", "self",
    "sell", "send", "shadow", "shallow", "shame", "shape", "sharp",
    "sheep", "shelf", "shell", "shield", "shift", "ship", "shoe",
    "shoot", "shore", "short", "shower", "shrink", "shut", "sick",
    "side", "sight", "silver", "sin", "sing", "sister", "sit", "skin",
    "skull", "sleep", "slide", "slim", "slow", "small", "smart",
    "smell", "smile", "smith", "smoke", "smooth", "snake", "snow",
    "soft", "son", "song", "sorrow", "soul", "sound", "south", "sow",
    "speed", "spell", "spider", "spring", "staff", "stall", "stand",
    "star", "stare", "start", "steal", "steam", "steel", "steep",
    "steer", "stem", "step", "stick", "still", "sting", "stink",
    "stir", "stock", "stone", "stool", "stop", "storm", "stove",
    "straw", "stream", "street", "strength", "stride", "strike",
    "string", "strong", "struggle", "stuff", "stump", "summer", "sun",
    "swallow", "swamp", "swan", "swear", "sweat", "sweet", "swell",
    "swift", "swim", "swing", "sword", "tale", "tall", "tame", "team",
    "tear", "tell", "ten", "thick", "thief", "thin", "thing", "think",
    "thorn", "thought", "thread", "three", "throat", "throw", "thumb",
    "thunder", "tide", "time", "tin", "toe", "tongue", "tool", "tooth",
    "town", "tree", "true", "trust", "truth", "turn", "twelve", "twin",
    "want", "war", "warm", "wash", "waste", "watch", "water", "wave",
    "way", "weak", "wealth", "weapon", "weather", "web", "wedding",
    "week", "weigh", "well", "west", "wet", "whale", "wheat", "wheel",
    "white", "wide", "wife", "wild", "will", "wind", "window", "wine",
    "wing", "winter", "wise", "wish", "with", "witness", "wolf",
    "woman", "wonder", "wood", "wool", "word", "work", "world", "worm",
    "worth", "wound", "wrist", "write", "wrong", "yard", "year",
    "yell", "yellow", "yield", "yoke", "young",
]

CURATED_LATIN = [
    "ability", "absent", "accept", "accident", "account", "accuse",
    "act", "action", "active", "actual", "addition", "address",
    "adjust", "admire", "admit", "advance", "adventure", "advice",
    "affair", "affect", "age", "agent", "agree", "aim", "allow",
    "amount", "ancient", "angel", "anger", "angle", "animal",
    "announce", "annual", "answer", "anxiety", "apparent", "appeal",
    "appear", "apply", "appoint", "army", "arrange", "arrest",
    "arrive", "art", "article", "assist", "assume", "attach",
    "attack", "attempt", "attend", "attention", "attract", "aunt",
    "author", "avoid", "balance", "ball", "band", "bank", "bar",
    "base", "battle", "beauty", "blame", "block", "board", "bond",
    "border", "bottle", "branch", "brave", "brief", "broad",
    "budget", "burden", "cabinet", "calculate", "campaign", "cancel",
    "capable", "capital", "captain", "capture", "carbon", "career",
    "carry", "case", "castle", "cause", "cease", "ceiling", "cell",
    "central", "certain", "chain", "chair", "challenge", "chamber",
    "champion", "chance", "channel", "chapter", "charge", "charm",
    "chase", "chief", "choice", "circle", "citizen", "civil",
    "claim", "class", "clear", "climate", "close", "cloud", "coach",
    "code", "college", "color", "column", "combine", "comfort",
    "command", "commerce", "commit", "common", "community", "company",
    "compare", "compete", "complete", "complex", "compose", "concern",
    "condition", "conduct", "confirm", "connect", "consider",
    "constant", "construct", "consume", "contact", "contain",
    "content", "contest", "continue", "contract", "control",
    "convert", "convince", "count", "country", "couple", "courage",
    "course", "court", "cover", "create", "credit", "crime",
    "crisis", "crown", "cruel", "culture", "cure", "current",
    "custom", "damage", "danger", "debt", "decide", "declare",
    "decline", "defeat", "defend", "define", "degree", "delay",
    "deliver", "demand", "deny", "depart", "depend", "describe",
    "design", "desire", "destroy", "detail", "determine", "develop",
    "device", "devote", "dignity", "direct", "discuss", "display",
    "distance", "distinct", "divide", "doctor", "doctrine", "domain",
    "doubt", "draft", "dress", "due", "duty", "economic", "education",
    "effect", "effort", "election", "element", "embrace",
    "emerge", "emotion", "empire", "employ", "enable", "encounter",
    "encourage", "enemy", "energy", "engage", "engine", "enhance",
    "enjoy", "enormous", "ensure", "enter", "entire", "entry",
    "envy", "equal", "error", "escape", "essence", "establish",
    "estate", "estimate", "event", "evident", "evil", "exact",
    "examine", "example", "excellent", "except", "exchange", "excuse",
    "execute", "exercise", "exist", "expand", "expect", "expense",
    "experience", "experiment", "expert", "explain", "express",
    "extend", "extent", "extreme", "face", "fact", "factor", "fail",
    "faith", "false", "familiar", "family", "famous", "fashion",
    "favor", "feature", "federal", "figure", "final", "finance",
    "firm", "fix", "flower", "force", "foreign", "forest", "form",
    "formal", "former", "fortune", "found", "foundation", "fruit",
    "fuel", "function", "fund", "future", "gallery", "garden",
    "general", "generous", "gentle", "glory", "govern", "government",
    "grace", "grade", "grain", "grand", "grant", "grave", "guard",
    "guide", "habit", "harbor", "honor", "horror", "host", "hour",
    "human", "humble", "idea", "ignore", "image", "imagine",
    "immediate", "immense", "impact", "import", "impose", "improve",
    "incident", "include", "increase", "indicate", "individual",
    "industry", "influence", "inform", "initial", "injury", "inner",
    "innocent", "insist", "inspire", "install", "instance", "instant",
    "instead", "interest", "internal", "interpret", "introduce",
    "invade", "invest", "invite", "involve", "issue", "item", "join",
    "journal", "journey", "judge", "justice", "labor", "language",
    "large", "launch", "layer", "league", "legal", "lesson", "level",
    "liberal", "liberty", "library", "license", "limit", "line",
    "link", "liquid", "list", "literature", "local", "logic", "lose",
    "loyal", "machine", "magic", "maintain", "major", "manage",
    "manner", "mansion", "margin", "market", "master", "material",
    "matter", "mature", "mayor", "measure", "medicine", "member",
    "memory", "mental", "mention", "message", "method", "military",
    "minister", "minor", "miracle", "mission", "model", "moderate",
    "modern", "modest", "moment", "moral", "motion", "mount", "move",
    "murder", "muscle", "mystery", "nation", "native", "natural",
    "nature", "navy", "necessary", "negotiate", "neutral", "noble",
    "normal", "note", "notice", "notion", "novel", "number",
    "object", "observe", "obtain", "obvious", "occasion", "occupy",
    "occur", "offend", "offer", "office", "opinion", "oppose",
    "option", "order", "ordinary", "origin", "ounce", "overall",
    "pace", "page", "pain", "pair", "palace", "panel", "parent",
    "part", "partial", "partner", "party", "passage", "passion",
    "patient", "pattern", "pause", "peace", "people", "perfect",
    "perform", "period", "permit", "person", "physical", "piece",
    "place", "plain", "plan", "plant", "platform", "please",
    "pleasure", "pledge", "plunge", "point", "poison", "policy",
    "polite", "popular", "portion", "position", "possess", "possible",
    "poverty", "power", "practice", "praise", "prayer", "precious",
    "predict", "prefer", "prepare", "present", "preserve", "press",
    "pressure", "pretend", "prevent", "price", "pride", "prince",
    "principal", "principle", "prison", "private", "prize", "problem",
    "proceed", "process", "produce", "profit", "program", "progress",
    "project", "promise", "promote", "proper", "property", "propose",
    "protect", "prove", "provide", "province", "public", "purchase",
    "pure", "purpose", "pursue", "quality", "quarter", "question",
    "race", "range", "rank", "rate", "ray", "reason", "receive",
    "record", "recover", "reduce", "reform", "refuse", "regard",
    "region", "regular", "reign", "reject", "relate", "release",
    "relief", "religion", "rely", "remain", "remark", "remedy",
    "remove", "render", "repair", "repeat", "replace", "report",
    "represent", "request", "require", "reserve", "resign", "resist",
    "resolve", "resource", "respond", "rest", "restore", "result",
    "retire", "reveal", "revenge", "revenue", "reverse", "review",
    "revolution", "rich", "rival", "river", "role", "route", "royal",
    "ruin", "rule", "sacred", "sacrifice", "saint", "sample",
    "satisfy", "save", "scene", "search", "season", "secret",
    "section", "secure", "select", "sense", "sentence", "separate",
    "series", "serious", "serve", "service", "session", "settle",
    "severe", "signal", "silent", "similar", "simple", "single",
    "situation", "social", "society", "soldier", "solid", "solution",
    "source", "space", "special", "spirit", "spread", "stable",
    "stage", "standard", "state", "station", "status", "strange",
    "strategy", "stress", "strict", "structure", "study", "style",
    "subject", "submit", "success", "suffer", "suggest", "suit",
    "superior", "supply", "support", "suppose", "supreme", "surface",
    "survive", "suspect", "symbol", "system", "table", "talent",
    "target", "task", "tax", "temple", "tend", "tension", "term",
    "territory", "terror", "test", "title", "total", "touch", "tour",
    "trace", "track", "trade", "tradition", "train", "transform",
    "transport", "travel", "treat", "treaty", "trial", "tribute",
    "triumph", "trouble", "trust", "union", "unique", "unit",
    "universe", "usual", "value", "variety", "vessel", "victory",
    "view", "village", "virtue", "vision", "visual", "vital", "voice",
    "volume", "vote", "wage", "venture",
]

CURATED_GREEK = [
    "academy", "acrobat", "aerobic", "aesthetic", "agony", "algorithm",
    "allegory", "allergy", "alphabet", "amphibian", "analogy", "analysis",
    "anatomy", "anchor", "anemia", "angel", "angle", "anomaly",
    "anonymous", "anthem", "antibiotic", "antique", "apathy", "apostle",
    "architect", "archive", "arctic", "aristocrat", "arithmetic",
    "aroma", "arthritis", "asteroid", "astronaut", "astronomy",
    "athlete", "atmosphere", "atom", "authentic", "automatic",
    "bacteria", "baptize", "barometer", "basic", "bible", "biography",
    "biology", "bishop", "blasphemy", "botany", "bronze", "bureaucracy",
    "calendar", "calligraphy", "catastrophe", "category", "cathedral",
    "catholic", "cemetery", "center", "ceramic", "character",
    "chi", "chlorine", "choir", "chord", "chromosome", "chronic",
    "chronology", "cinema", "clergy", "climate", "comedy", "comet",
    "cosmetic", "cosmic", "cosmos", "crisis", "criterion", "critic",
    "crystal", "cycle", "cylinder", "cynic", "democracy", "demon",
    "dermatology", "diagnosis", "diagram", "dialect", "dialogue",
    "diameter", "diamond", "diaphragm", "diet", "dilemma", "dinosaur",
    "diploma", "diplomat", "disc", "drama", "dynamic", "dynasty",
    "echo", "eclipse", "ecology", "economy", "elastic", "electric",
    "electron", "element", "elephant", "ellipse", "emblem", "embryo",
    "emphasis", "encyclopedia", "endemic", "energy", "enigma",
    "enthusiasm", "enzyme", "ephemeral", "epidemic", "epilepsy",
    "episode", "epitome", "epoch", "erotic", "ethics", "ethnic",
    "eulogy", "euphoria", "euthanasia", "evangelist", "exotic",
    "fantasy", "galaxy", "genesis", "genetic", "genius", "genre",
    "geography", "geology", "geometry", "giant", "glycerin",
    "grammar", "graph", "gymnasium", "gynecology", "harmony",
    "hegemony", "helicopter", "hemisphere", "hemorrhage", "hepatitis",
    "heresy", "hero", "hierarchy", "hippopotamus", "history",
    "hologram", "homonym", "horizon", "hormone", "horoscope",
    "hydraulic", "hydrogen", "hygiene", "hymn", "hypnosis",
    "hypothesis", "hysteria", "icon", "idea", "ideology", "idiot",
    "idol", "irony", "kinetic", "labyrinth", "lexicon", "liturgy",
    "logic", "lyric", "machine", "magnet", "mania", "marathon",
    "martyr", "mathematics", "mechanic", "megaphone", "melody",
    "membrane", "metabolism", "metaphor", "meteor", "method",
    "metropolis", "microbe", "microscope", "mimic", "mineral",
    "miracle", "misanthrope", "monarch", "monastery", "monopoly",
    "morphology", "museum", "music", "mystery", "myth", "narcissist",
    "nausea", "nectar", "nemesis", "neurology", "nomad", "nostalgia",
    "nymph", "ocean", "odyssey", "oligarchy", "olympics", "omega",
    "opera", "optical", "oracle", "orchestra", "organ", "organic",
    "organism", "orient", "orphan", "orthodox", "ostracize", "oxygen",
    "pagan", "palindrome", "pamphlet", "pandemic", "panic",
    "panorama", "parable", "paradigm", "paradise", "paradox",
    "paragraph", "parallel", "parasite", "parenthesis", "pathology",
    "patriot", "pedagogy", "pentagon", "period", "periphery",
    "petroleum", "phantom", "pharmacy", "phenomenon", "philanthropy",
    "philosophy", "phobia", "phone", "photograph", "phrase", "physics",
    "planet", "plasma", "plastic", "plethora", "pneumonia", "poem",
    "polemic", "police", "policy", "politics", "polygon", "polytheism",
    "practice", "pragmatic", "prism", "problem", "program",
    "prologue", "prophecy", "protagonist", "protein", "psalm",
    "pseudonym", "psychiatry", "psychology", "pyramid", "python",
    "rhapsody", "rhetoric", "rhinoceros", "rhythm", "sarcasm",
    "satellite", "scandal", "scenario", "scene", "schizophrenia",
    "scholar", "school", "siren", "skeleton", "skeptic", "somatic",
    "sophisticated", "sphere", "stadium", "static", "stellar",
    "stereotype", "stigma", "stoic", "stomach", "strategy",
    "syllable", "symbol", "symmetry", "sympathy", "symphony",
    "symptom", "synagogue", "syndrome", "synonym", "syntax",
    "synthesis", "system", "talent", "technical", "technology",
    "telegram", "telephone", "telescope", "temperature", "theater",
    "theme", "theology", "theory", "therapy", "thermal", "thesis",
    "thorax", "throne", "tone", "topic", "topography", "toxic",
    "tragedy", "trauma", "trilogy", "triumph", "trophy", "tropic",
    "tyrant", "utopia", "zodiac", "zone", "zoology",
]

CURATED_OTHER = [
    # Arabic
    "admiral", "alchemy", "alcohol", "alcove", "algebra", "algorithm",
    "almanac", "amber", "arsenal", "artichoke", "assassin", "average",
    "azimuth", "azure", "cable", "caliber", "camel", "candy", "carafe",
    "carat", "caravan", "check", "checkmate", "chemistry", "cipher",
    "coffee", "cotton", "crimson", "elixir", "emir", "garrison",
    "gauze", "gazelle", "genie", "ghoul", "giraffe", "guitar",
    "harem", "hazard", "henna", "jar", "jasmine", "lemon", "lilac",
    "lime", "lute", "magazine", "mask", "mattress", "monsoon",
    "mosque", "mummy", "nadir", "orange", "safari", "saffron",
    "sequin", "sherbet", "sofa", "spinach", "sugar", "sultan",
    "sumac", "syrup", "taboo", "talisman", "tambourine", "tariff",
    "zenith", "zero",
    "pariah", "pepper", "punch", "rajah", "rupee", "sandal", "sapphire", "shampoo",
    "shawl", "sugar", "thug", "toddy", "typhoon", "veranda", "yoga",
    "zen", "swastika", "loot", "mandarin", "mantra", "mogul", "musk",
    "nirvana", "pajama", "avatar", "bandana", "bangle", "bazaar",
    "bungalow", "cashmere", "catamaran", "cheetah", "chintz", "chit",
    "cot", "cowrie", "crimson", "dinghy", "dungaree", "guru", "jungle",
    "juggernaut", "karma", "khaki", "lacquer",
]

CURATED_SANSKRIT = [
    "avatar", "guru", "karma", "yoga", "nirvana", "jungle", "bandit", "shampoo",
    "pajama", "punch", "bandana", "bangle", "bazaar", "bungalow", "cashmere",
    "catamaran", "cheetah", "chintz", "chit", "cot", "cowrie", "dinghy", 
    "dungaree", "juggernaut", "khaki", "lacquer", "loot", "mandarin", "mantra",
    "mogul", "musk", "pariah", "pepper", "rajah", "rupee", "sandal", "sapphire",
    "shawl", "sugar", "thug", "toddy", "typhoon", "veranda", "swastika",
    "pali", "sutra", "dharma", "veda", "pundit", "loot", "thug", "cot",
    "buck", "chutney", "cot", "cushion", "dinghy", "dungaree", "ghat",
    "gymkhana", "juggernaut", "jute", "khaki", "mulligatawny", "nawab",
    "pukka", "pyjamas", "sahib", "seapoy", "shampoo", "thug", "toddy",
    "verandah", "yoga", "zenana",
]

CURATED_PIE = [
    "father", "mother", "brother", "sister", "son", "daughter", "water",
    "new", "three", "two", "one", "night", "day", "eye", "ear", "nose",
    "mouth", "tooth", "bone", "heart", "blood", "hand", "foot", "knee",
    "name", "star", "sun", "moon", "fire", "wind", "snow", "tree", "seed",
    "cow", "horse", "dog", "wolf", "bear", "honey", "mead", "ax", "door",
    "road", "town", "word", "god", "life", "death", "red", "white", "black",
    "long", "short", "wide", "deep", "hot", "cold", "sweet", "bitter",
    "full", "thin", "heavy", "light", "old", "young", "good", "bad",
    "man", "woman", "child", "king", "slave", "war", "peace", "fear",
]

CURATED_OTHER = [
    # Arabic
    "banana", "barbecue", "canoe", "chocolate", "cigar", "cocoa",
    "condor", "hammock", "hurricane", "igloo", "kayak", "ketchup",
    "cookie", "landscape", "maize", "moose", "penguin", "potato",
    "ranch", "savanna", "skunk", "squash", "tobacco", "toboggan",
    "tomato", "tornado",
]

CURATED_LISTS = {
    "Germanic": CURATED_GERMANIC,
    "Latin": CURATED_LATIN,
    "Greek": CURATED_GREEK,
    "Sanskrit": CURATED_SANSKRIT,
    "PIE": CURATED_PIE,
    "Other": CURATED_OTHER,
}


def supplement_with_curated(
    word_to_class: dict[str, str],
    min_per_class: int = MIN_PER_CLASS,
) -> dict[str, str]:
    """
    Add curated words for any class that is underrepresented.
    Also useful to ensure known-correct labels for important words.
    """
    print("\n[Step 4] Supplementing with curated word lists...")

    # Count current distribution
    counts = Counter(word_to_class.values())
    for cls in ORIGIN_CLASSES:
        print(f"  {cls}: {counts.get(cls, 0)} words (before supplement)")

    added = 0
    overridden = 0
    for origin_class, word_list in CURATED_LISTS.items():
        for word in word_list:
            w = word.strip().lower()
            if not w.isalpha() or len(w) < 3:
                continue
            if w not in word_to_class:
                word_to_class[w] = origin_class
                added += 1
            elif word_to_class[w] != origin_class:
                # Override etymwn label with curated (more trustworthy)
                word_to_class[w] = origin_class
                overridden += 1

    counts_after = Counter(word_to_class.values())
    print(f"\n  Added {added} new curated words")
    print(f"  Overrode {overridden} etymwn labels with curated labels")
    for cls in ORIGIN_CLASSES:
        print(f"  {cls}: {counts_after.get(cls, 0)} words (after supplement)")

    return word_to_class


# ──────────────────────────────────────────────────────────────
#  Step 5: Balance & sample
# ──────────────────────────────────────────────────────────────
def balance_dataset(
    word_to_class: dict[str, str],
    target_per_class: int = TARGET_PER_CLASS,
    max_per_class: int = MAX_PER_CLASS,
) -> pd.DataFrame:
    """
    Balance the dataset by sampling down oversized classes.
    Curated words are always kept; only etymwn-sourced words are trimmed.
    """
    print(f"\n[Step 5] Balancing dataset (target ~{target_per_class}/class)...")

    # Build a set of all curated words for priority
    curated_set = set()
    for word_list in CURATED_LISTS.values():
        for w in word_list:
            curated_set.add(w.strip().lower())

    rows = []
    for cls in ORIGIN_CLASSES:
        all_words = [w for w, c in word_to_class.items() if c == cls]

        # Separate curated (priority) from etymwn-only words
        curated_words = [w for w in all_words if w in curated_set]
        etymwn_words = [w for w in all_words if w not in curated_set]
        random.shuffle(etymwn_words)

        # Always keep curated; fill remaining with etymwn
        remaining_slots = max(0, max_per_class - len(curated_words))
        selected = curated_words + etymwn_words[:remaining_slots]

        for w in selected:
            rows.append({"word": w, "origin_class": cls})

        print(f"  {cls}: {len(selected)} words ({len(curated_words)} curated + {min(len(etymwn_words), remaining_slots)} etymwn)")

    df = pd.DataFrame(rows)
    print(f"\n  Total dataset size: {len(df):,} words")

    return df


# ──────────────────────────────────────────────────────────────
#  Step 6: Train/Test split
# ──────────────────────────────────────────────────────────────
def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split."""
    print(f"\n[Step 6] Splitting into train/test ({TRAIN_TEST_SPLIT:.0%} / {1-TRAIN_TEST_SPLIT:.0%})...")

    train_df, test_df = train_test_split(
        df,
        test_size=1 - TRAIN_TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=df["origin_class"],
    )

    print(f"  Train: {len(train_df):,} words")
    print(f"  Test:  {len(test_df):,} words")

    return train_df, test_df


# ──────────────────────────────────────────────────────────────
#  Step 7: Stats & save
# ──────────────────────────────────────────────────────────────
def save_and_report(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Save CSVs and generate summary statistics."""
    print("\n[Step 7] Saving datasets and statistics...")

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    df.to_csv(ORIGIN_DATASET, index=False)
    train_df.to_csv(TRAIN_DATASET, index=False)
    test_df.to_csv(TEST_DATASET, index=False)

    print(f"  ✓ {ORIGIN_DATASET.name}: {len(df):,} rows")
    print(f"  ✓ {TRAIN_DATASET.name}: {len(train_df):,} rows")
    print(f"  ✓ {TEST_DATASET.name}: {len(test_df):,} rows")

    # Build statistics report
    lines = []
    lines.append("=" * 50)
    lines.append("  Etymology Dataset Statistics")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Total words:  {len(df):,}")
    lines.append(f"Train split:  {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    lines.append(f"Test split:   {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    lines.append("")
    lines.append("Class Distribution (Full Dataset):")
    lines.append("-" * 40)

    for cls in ORIGIN_CLASSES:
        count = len(df[df["origin_class"] == cls])
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        lines.append(f"  {cls:<10} {count:>5}  ({pct:5.1f}%) {bar}")

    lines.append("")
    lines.append("Sample words per class:")
    lines.append("-" * 40)

    for cls in ORIGIN_CLASSES:
        words = df[df["origin_class"] == cls]["word"].tolist()
        sample = random.sample(words, min(10, len(words)))
        lines.append(f"  {cls}: {', '.join(sample)}")

    lines.append("")

    # Spot-check known words
    known = {
        "house": "Germanic", "mother": "Germanic", "water": "Germanic",
        "government": "Latin", "culture": "Latin", "justice": "Latin",
        "biology": "Greek", "philosophy": "Greek", "democracy": "Greek",
        "algebra": "Other", "coffee": "Other", "tsunami": "Other",
    }
    lines.append("Spot-check (known origins):")
    lines.append("-" * 40)
    for word, expected in known.items():
        row = df[df["word"] == word]
        if len(row) > 0:
            actual = row.iloc[0]["origin_class"]
            status = "✓" if actual == expected else f"✗ (got {actual})"
            lines.append(f"  {word:<15} expected={expected:<10} {status}")
        else:
            lines.append(f"  {word:<15} NOT IN DATASET")

    report = "\n".join(lines)

    # Save stats file
    with open(DATASET_STATS, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  ✓ {DATASET_STATS.name}")
    print(f"\n{report}")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Phase 1: Etymology Dataset Builder")
    print("=" * 60)

    # Step 1: Parse etymwn.tsv
    word_origins = parse_etymwn(ETYMWN_FILE)

    # Step 2: Normalize origins
    word_to_class = normalize_origins(word_origins)

    # Step 3: Clean & filter
    word_to_class = clean_words(word_to_class)

    # Step 4: Supplement with curated lists
    word_to_class = supplement_with_curated(word_to_class)

    # Step 5: Balance dataset
    df = balance_dataset(word_to_class)

    # Step 6: Train/Test split
    train_df, test_df = split_dataset(df)

    # Step 7: Save & report
    save_and_report(df, train_df, test_df)

    print(f"\n{'=' * 60}")
    print(f"  Phase 1 Complete!")
    print(f"  Dataset ready at: {DATA_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
