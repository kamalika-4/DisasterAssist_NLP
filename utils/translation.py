```python
"""
DisasterAssist NLP - Multilingual Translation Engine

Uses MarianMT sentence-level translation models.

Supported:
Tamil  -> English
Hindi  -> English
Spanish -> English
French -> English

No predefined emergency sentences.
No GoogleTranslator.
"""

import re
import torch
from transformers import MarianMTModel, MarianTokenizer


# =========================================================
# MODEL NAMES
# =========================================================

MODEL_NAMES = {
    "ta": "Helsinki-NLP/opus-mt-tc-big-en-ta",
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
}


LANG_NAMES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
}


# =========================================================
# MODEL CACHE
# =========================================================

_tokenizers = {}
_models = {}


def _load_model(source_lang):
    """
    Load the MarianMT model for the selected source language.
    The model is loaded only once and then kept in memory.
    """

    if source_lang not in MODEL_NAMES:
        raise ValueError(
            f"Unsupported language: {source_lang}"
        )

    if source_lang not in _tokenizers:

        model_name = MODEL_NAMES[source_lang]

        print(
            f"Loading MarianMT model for "
            f"{LANG_NAMES[source_lang]}..."
        )

        _tokenizers[source_lang] = MarianTokenizer.from_pretrained(
            model_name
        )

        _models[source_lang] = MarianMTModel.from_pretrained(
            model_name
        )

        _models[source_lang].eval()

        print(
            f"{LANG_NAMES[source_lang]} translation model loaded."
        )

    return (
        _tokenizers[source_lang],
        _models[source_lang]
    )


# =========================================================
# LANGUAGE DETECTION
# =========================================================

def detect_language_simple(text):
    """
    Basic Unicode-script language detection.

    Tamil and Hindi have unique scripts.

    Latin-script languages are handled as English by default.
    """

    if not text or not text.strip():
        return "en"

    text = text.strip()

    tamil_count = len(
        re.findall(r"[\u0B80-\u0BFF]", text)
    )

    hindi_count = len(
        re.findall(r"[\u0900-\u097F]", text)
    )

    total = max(len(text), 1)

    if tamil_count / total > 0.10:
        return "ta"

    if hindi_count / total > 0.10:
        return "hi"

    return "en"


# =========================================================
# TRANSLATE TO ENGLISH
# =========================================================

def translate_to_english(text, source_lang="auto"):
    """
    Translate the complete user message into English.

    No predefined sentences are used.
    """

    if not text or not text.strip():
        return ""

    text = text.strip()

    # Automatic language detection
    if source_lang == "auto":
        source_lang = detect_language_simple(text)

    # Already English
    if source_lang == "en":
        return text

    if source_lang not in MODEL_NAMES:
        raise ValueError(
            f"Unsupported source language: {source_lang}"
        )

    tokenizer, model = _load_model(source_lang)

    # Tokenize the complete message
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Generate complete translation
    with torch.no_grad():

        translated_tokens = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )

    # Decode complete sentence
    translated = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return translated.strip()


# =========================================================
# ENGLISH -> OTHER LANGUAGES
# =========================================================

def translate_from_english(text, target_lang):
    """
    Translate English into a supported language.

    This function is kept for compatibility with the
    existing DisasterAssist application.
    """

    if not text or not text.strip():
        return "", 0.0

    text = text.strip()

    # Already English
    if target_lang == "en":
        return text, 1.0

    # MarianMT models above are primarily configured
    # for source-language -> English translation.
    # Therefore this function is not used for those models.
    return text, 0.0
```
