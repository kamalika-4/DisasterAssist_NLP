# 🚨 DisasterAssist NLP
### Multilingual Emergency Message Understanding & Response System

A full Streamlit application demonstrating an end-to-end NLP/ML pipeline for
disaster-response message understanding across **English, Tamil, Hindi,
Spanish, and French**.

## Modules Covered
- Machine Translation (emergency-domain lexicon-based, EN⇄TA/HI/ES/FR)
- Multilingual NLP (language detection, script + statistical)
- Text Classification (TF-IDF + Logistic Regression / Random Forest — disaster type & severity)
- Speech Processing (STT via Google Speech Recognition, TTS via gTTS)
- Named Entity Recognition (spaCy `en_core_web_sm` + custom disaster/casualty regex)
- Question Answering (TF-IDF cosine-similarity retrieval over an emergency knowledge base)
- Chatbot (contextual multi-turn assistant combining every module above)
- Market Basket Analysis (Apriori / association rules via `mlxtend` over relief-supply requests)
- NLP Preprocessing & Similarity (cleaning, tokenization, lemmatization, Jaccard & cosine similarity)

## Pages
| Page | Description |
|---|---|
| **Overview** (`app.py`) | KPIs, ETL pipeline walkthrough, full EDA (language/disaster/severity distributions, message-length histogram, heatmap) |
| **Live Emergency Assistant** | Paste/select an emergency message → full pipeline output with confidence scores |
| **NLP Analysis** | Interactive language detection, translation, preprocessing, similarity, NER workbench |
| **Speech Processing** | Record/upload audio → speech-to-text transcription → emergency analysis; text-to-speech playback |
| **Analytics** | Real model metrics: accuracy/F1/precision/recall, confusion matrices, per-class report |
| **Emergency Chatbot** | Conversational assistant with session context (last disaster type/severity/location) |
| **Market Basket Analysis** | Apriori association rules over relief-resource request transactions, with an interactive rule explorer |

## Run locally
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # or install wheel directly, see below
python utils/data_generator.py            # generate dataset (auto-runs on first app launch too)
streamlit run app.py
```

If `python -m spacy download en_core_web_sm` cannot reach the spaCy CDN in your
environment, install the model wheel directly from GitHub releases:
```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

## Notes on "real" processing
- All models (classification, QA retrieval, association rules) are **trained/computed
  at runtime** from a generated dataset (`utils/data_generator.py`) — not hard-coded outputs.
- Metrics (accuracy, F1, confusion matrices) are computed on a genuine held-out
  20% test split, cached with `joblib` so the app doesn't retrain on every rerun.
- Translation uses a curated multilingual emergency lexicon with longest-phrase-match
  substitution and a real coverage-based confidence score, since this environment has
  no outbound network access to commercial translation APIs.
- Speech-to-text uses Google Speech Recognition via SpeechRecognition.
  Text-to-speech (gTTS) requires outbound internet to Google's TTS endpoint and will
  show a clear fallback message if unavailable.

## Folder structure
```
DisasterAssist_NLP/
├── app.py                     # Overview / EDA / ETL (Home)
├── pages/
│   ├── 1_Live_Emergency_Assistant.py
│   ├── 2_NLP_Analysis.py
│   ├── 3_Speech_Processing.py
│   ├── 4_Analytics.py
│   ├── 5_Emergency_Chatbot.py
│   └── 6_Market_Basket_Analysis.py
├── utils/
│   ├── data_generator.py      # synthetic multilingual dataset + MBA transactions
│   ├── preprocessing.py       # language detection, cleaning, tokenization, similarity
│   ├── translation.py         # lexicon-based multilingual translation
│   ├── ner_utils.py           # spaCy NER + custom disaster/casualty extraction
│   ├── classification.py      # disaster-type & severity classifiers + metrics
│   ├── qa_engine.py           # TF-IDF retrieval QA over emergency knowledge base
│   ├── chatbot.py             # orchestrates the full pipeline for chat/assistant
│   ├── speech_utils.py        # STT (Google Speech Recognition) / TTS (gTTS) / signal analysis
│   └── app_config.py          # shared CSS/theme helpers
├── data/emergency_messages.csv
├── models/                    # cached trained models (joblib)
└── requirements.txt
```
