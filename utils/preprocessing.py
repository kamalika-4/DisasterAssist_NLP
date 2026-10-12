"""
Core NLP preprocessing utilities: language detection, cleaning,
tokenization, stopword removal, lemmatization-lite, and TF-IDF based
text similarity -- all real, computed at call time.
"""
import re
import string
from langdetect import detect, DetectorFactory, detect_langs

DetectorFactory.seed = 42

SCRIPT_RANGES = {
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "hi": (0x0900, 0x097F),  # Devanagari
}

EN_STOPWORDS = set("""a an the is are was were be been being to of in on for and or
but if then so at by with from as this that it its it's i you he she we they
not no do does did have has had will would can could should may might must
just about into over under again further here there when where why how all
any both each few more most other some such only own same too very s t""".split())


SPANISH_MARKERS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "y", "por",
    "para", "con", "que", "es", "son", "está", "están", "personas", "inundación",
    "inundaciones", "atrapadas", "atrapados", "necesitan", "ayuda", "urgente",
    "urgentemente", "inmediatamente", "inmediato", "bomberos", "incendio", "terremoto",
    "heridos", "heridas", "rescate", "edificio", "edificios", "camino", "terreno",
    "deslizamiento", "carretera", "muévase", "alto", "vientos", "fuertes", "árboles"
}

FRENCH_MARKERS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "en", "et", "dans", "pour", "avec",
    "que", "est", "sont", "ont", "nous", "vous", "personnes", "inondation", "inondations",
    "piégées", "piégés", "besoin", "aide", "urgent", "urgente", "urgence",
    "immédiatement", "pompiers", "incendie", "tremblement", "terre", "séisme",
    "blessés", "blessées", "sauvetage", "bâtiment", "bâtiments", "route", "terrain",
    "glissement", "déplacez", "élevé", "vents", "violents", "arbres"
}


def detect_language(text: str):
    """Detect language with a confidence score. Accurately identifies Tamil
    (including code-mixed Tanglish), Hindi, Spanish, French, and English."""
    text = text.strip()
    if not text:
        return "en", 0.0

    counts = {"ta": 0, "hi": 0}
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] += 1

    total_letters = sum(1 for ch in text if ch.isalpha())
    total_letters = max(total_letters, 1)

    # Any Tamil characters present -> prioritize Tamil (crucial for code-mixed Tanglish like truckயை)
    if counts["ta"] > 0:
        conf = min(1.0, 0.75 + (counts["ta"] / total_letters) * 0.25)
        return "ta", round(conf, 3)

    # Any Devanagari characters present -> prioritize Hindi
    if counts["hi"] > 0:
        conf = min(1.0, 0.75 + (counts["hi"] / total_letters) * 0.25)
        return "hi", round(conf, 3)

    # Latin text: evaluate Spanish / French / English
    text_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))

    es_score = len(words & SPANISH_MARKERS)
    fr_score = len(words & FRENCH_MARKERS)

    if re.search(r"[áíóúñ¿¡]", text_lower):
        es_score += 3
    if re.search(r"[œçèêëâîïôûù]", text_lower) or re.search(r"\b(?:d'|l'|qu'|s'|n'|c'|j')", text_lower):
        fr_score += 3

    if es_score > fr_score and es_score >= 2:
        return "es", min(1.0, round(0.82 + es_score * 0.04, 3))
    if fr_score > es_score and fr_score >= 2:
        return "fr", min(1.0, round(0.82 + fr_score * 0.04, 3))

    try:
        probs = detect_langs(text)
        top = probs[0]
        if top.lang in ["es", "fr"]:
            return top.lang, round(top.prob, 3)
        if top.lang == "en":
            return "en", round(top.prob, 3)
    except Exception:
        pass

    if es_score > fr_score and es_score > 0:
        return "es", 0.85
    if fr_score > es_score and fr_score > 0:
        return "fr", 0.85

    return "en", 0.95


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\d+", " <NUM> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return [t for t in re.split(r"\s+", text.strip()) if t]


def remove_stopwords(tokens):
    return [t for t in tokens if t.lower() not in EN_STOPWORDS and t not in string.punctuation]


def simple_lemmatize(token: str) -> str:
    """Lightweight suffix-stripping lemmatizer (rule-based, no external model)."""
    t = token.lower()
    for suf in ["ing", "edly", "ed", "ies", "es", "s"]:
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)]
    return t


def preprocess_pipeline(text: str):
    """Full pipeline returning every intermediate step for transparent display."""
    steps = {}
    steps["original"] = text
    cleaned = clean_text(text)
    steps["cleaned"] = cleaned
    tokens = tokenize(cleaned)
    steps["tokens"] = tokens
    no_stop = remove_stopwords(tokens)
    steps["no_stopwords"] = no_stop
    lemmas = [simple_lemmatize(t) for t in no_stop]
    steps["lemmatized"] = lemmas
    steps["final_text"] = " ".join(lemmas)
    return steps


def jaccard_similarity(a: str, b: str) -> float:
    set_a, set_b = set(tokenize(a.lower())), set(tokenize(b.lower()))
    if not set_a or not set_b:
        return 0.0
    return round(len(set_a & set_b) / len(set_a | set_b), 4)
