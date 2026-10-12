import os
import sys

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from utils.app_config import inject_css, page_header
from utils.preprocessing import (
    detect_language,
    preprocess_pipeline,
    jaccard_similarity
)
from utils.translation import (
    translate_to_english,
    translate_from_english,
    LANG_NAMES
)
from utils.ner_utils import extract_entities


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NLP Analysis",
    page_icon="🔬",
    layout="wide"
)

inject_css()

page_header(
    "🔬 NLP Analysis Workbench",
    "Language detection · preprocessing · translation · similarity · NER — all computed live"
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🈯 Language & Translation",
        "🧹 Preprocessing Pipeline",
        "🧮 Text Similarity",
        "🏷️ Named Entity Recognition"
    ]
)


# =========================================================
# TAB 1 — LANGUAGE DETECTION & TRANSLATION
# =========================================================

with tab1:

    st.markdown("### Language Detection & Machine Translation")

    text = st.text_input(
        "Enter text in English / Tamil / Hindi / Spanish / French",
        "வெள்ளம் காரணமாக மக்கள் சிக்கியுள்ளனர், உதவி தேவை"
    )

    if text:

        # Language detection
        lang, conf = detect_language(text)

        st.success(
            f"Detected language: "
            f"**{LANG_NAMES.get(lang, lang)}** "
            f"(confidence: {conf:.1%})"
        )

        # Translation to English
        try:
            translated, coverage = translate_to_english(
                text,
                lang
            )

            c1, c2 = st.columns(2)

            c1.text_area(
                "Original",
                text,
                height=90,
                disabled=True
            )

            c2.text_area(
                "Translated to English",
                translated,
                height=90,
                disabled=True
            )

            st.markdown(f"**Translated to English:** {translated}")
        except Exception as e:
            st.error(f"⚠️ Translation Error: {e}")

        # English → target language
        st.markdown(
            "##### 🔁 Translate English guidance into another language"
        )

        target = st.selectbox(
            "Target language",
            ["ta", "hi", "es", "fr"],
            format_func=lambda x: LANG_NAMES[x]
        )

        en_sample = st.text_input(
            "English text to translate",
            "Move to higher ground immediately, help is on the way"
        )

        try:
            out, cov2 = translate_from_english(
                en_sample,
                target
            )

            st.info(
                f"**{LANG_NAMES[target]} translation:** {out}"
            )
        except Exception as e:
            st.error(f"⚠️ Translation Error: {e}")


# =========================================================
# TAB 2 — PREPROCESSING PIPELINE
# =========================================================

with tab2:

    st.markdown("### Preprocessing Pipeline")

    txt2 = st.text_area(
        "Enter English (or translated) emergency text",
        "Heavy flooding in Chennai! Water level is RISING fast, "
        "5 people trapped near the bridge."
    )

    if txt2:

        steps = preprocess_pipeline(txt2)

        # Original
        st.write(
            "**1. Original:**",
            steps["original"]
        )

        # Cleaned
        st.write(
            "**2. Cleaned "
            "(lowercasing handled at token level, "
            "punctuation/URLs/numbers stripped):**",
            steps["cleaned"]
        )

        # Tokenization
        st.write("**3. Tokenized:**")

        st.code(
            str(steps["tokens"])
        )

        # Stopword removal
        st.write("**4. Stopwords removed:**")

        st.code(
            str(steps["no_stopwords"])
        )

        # Lemmatization
        st.write(
            "**5. Lemmatized "
            "(rule-based suffix stripping):**"
        )

        st.code(
            str(steps["lemmatized"])
        )

        # Final feature string
        st.write(
            "**6. Final feature string used for ML models:**"
        )

        st.success(
            steps["final_text"]
        )

        # Token frequency
        wc = (
            pd.Series(steps["lemmatized"])
            .value_counts()
            .reset_index()
        )

        wc.columns = [
            "Token",
            "Count"
        ]

        if len(wc):

            fig = px.bar(
                wc,
                x="Token",
                y="Count",
                title="Token Frequency After Preprocessing"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# TAB 3 — TEXT SIMILARITY
# =========================================================

with tab3:

    st.markdown("### Text Similarity Engine")

    st.caption(
        "Compares two emergency messages using "
        "Jaccard similarity (token overlap) and "
        "TF-IDF cosine similarity (semantic weighting)."
    )

    ca, cb = st.columns(2)

    text_a = ca.text_area(
        "Message A",
        "Flood water rising near Chennai, "
        "people trapped, need boat rescue"
    )

    text_b = cb.text_area(
        "Message B",
        "Heavy flooding in Chennai, "
        "families stuck, urgent rescue needed"
    )

    if text_a and text_b:

        # Jaccard similarity
        jac = jaccard_similarity(
            text_a,
            text_b
        )

        # TF-IDF
        vec = TfidfVectorizer()

        tfidf = vec.fit_transform(
            [text_a, text_b]
        )

        # Cosine similarity
        cos_sim = cosine_similarity(
            tfidf[0],
            tfidf[1]
        )[0][0]

        # Display metrics
        m1, m2 = st.columns(2)

        m1.metric(
            "Jaccard Similarity",
            f"{jac:.3f}"
        )

        m2.metric(
            "TF-IDF Cosine Similarity",
            f"{cos_sim:.3f}"
        )

        st.progress(
            min(float(cos_sim), 1.0)
        )

        # Interpretation
        if cos_sim > 0.5:

            st.success(
                "These messages likely describe the "
                "**same incident** — candidate for "
                "deduplication/merging."
            )

        else:

            st.info(
                "These messages likely describe "
                "**different incidents**."
            )


# =========================================================
# TAB 4 — NAMED ENTITY RECOGNITION
# =========================================================

with tab4:

    st.markdown("### Named Entity Recognition")

    st.caption(
        "Extracts disaster-related entities such as "
        "locations, dates, organizations, disaster terms, "
        "and casualty counts."
    )

    txt3 = st.text_area(
        "Enter English emergency text for NER",
        "Ravi from Chennai reported a building collapse "
        "this morning, 4 people injured, ambulance requested"
    )

    if txt3:

        # Extract entities
        entities, doc = extract_entities(
            txt3
        )

        # -------------------------------------------------
        # Entity table
        # -------------------------------------------------

        st.markdown(
            "#### 🔎 Detected Entities"
        )

        rows = [
            {
                "Entity Type": key,
                "Value": value
            }
            for key, values in entities.items()
            for value in values
        ]

        if rows:

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No entities detected."
            )

        # -------------------------------------------------
        # Entity metrics
        # -------------------------------------------------

        st.markdown(
            "#### 📊 Entity Summary"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "📍 Locations",
            len(
                entities.get(
                    "LOCATION",
                    []
                )
            )
        )

        c2.metric(
            "📅 Dates",
            len(
                entities.get(
                    "DATE",
                    []
                )
            )
        )

        c3.metric(
            "🏢 Organizations",
            len(
                entities.get(
                    "ORG",
                    []
                )
            )
        )

        c4.metric(
            "🚨 Disaster Terms",
            len(
                entities.get(
                    "DISASTER_TERMS",
                    []
                )
            )
        )

        # -------------------------------------------------
        # Casualty information
        # -------------------------------------------------

        casualties = entities.get(
            "CASUALTY_COUNTS",
            []
        )

        if casualties:

            st.markdown(
                "#### 🚑 Casualty / Impact Information"
            )

            for casualty in casualties:

                st.error(
                    f"Detected: {casualty}"
                )

        # -------------------------------------------------
        # Disaster terms
        # -------------------------------------------------

        disaster_terms = entities.get(
            "DISASTER_TERMS",
            []
        )

        if disaster_terms:

            st.markdown(
                "#### ⚠️ Disaster Terms"
            )

            st.write(
                ", ".join(disaster_terms)
            )

        # -------------------------------------------------
        # NER explanation
        # -------------------------------------------------

        st.info(
            "NER identifies important information from an "
            "emergency message. Rule-based extraction is "
            "used in this environment because the spaCy "
            "native extension is blocked by Windows "
            "Application Control."
        )