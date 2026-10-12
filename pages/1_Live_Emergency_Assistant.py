import os, sys
import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.app_config import inject_css, page_header, severity_badge
from utils.chatbot import process_emergency_message

st.set_page_config(page_title="Live Emergency Assistant", page_icon="🆘", layout="wide")
inject_css()
page_header("🆘 Live Emergency Assistant", "End-to-end pipeline: language detection → translation → "
            "classification → NER → guidance retrieval")

SAMPLES = {
    "English — Flood": "Heavy flooding in Chennai, water level rising fast, 6 people trapped on rooftop",
    "Tamil — Fire": "சென்னையில் தீ விபத்து, புகை அதிகமாக உள்ளது, தயவுசெய்து தீயணைப்பு படையை அனுப்பவும்",
    "Hindi — Earthquake": "मुंबई में भूकंप आया, इमारतें हिल रही हैं, कई लोग घायल हुए हैं, तुरंत मदद चाहिए",
    "Spanish — Medical": "Persona en Chennai tiene dolor en el pecho, necesita ambulancia inmediatamente",
    "French — Accident": "Accident de la route près de Bangalore, plusieurs véhicules impliqués, aide nécessaire",
}

st.markdown("#### ✍️ Enter or select an emergency message")
c1, c2 = st.columns([3, 1])
with c2:
    sample_choice = st.selectbox("Try a sample message", ["-- custom --"] + list(SAMPLES.keys()))
with c1:
    default_text = SAMPLES.get(sample_choice, "")
    user_text = st.text_area("Emergency message (any of the 5 supported languages)",
                              value=default_text, height=100,
                              placeholder="e.g. Heavy flooding near Madurai, elderly couple trapped inside house")

analyze = st.button("🚀 Analyze Emergency Message", type="primary")

if "assist_session" not in st.session_state:
    st.session_state.assist_session = {}

if analyze and user_text.strip():
    with st.spinner("Running full NLP pipeline..."):
        result = process_emergency_message(user_text, st.session_state.assist_session)
    st.session_state["last_result"] = result

if "last_result" in st.session_state:
    r = st.session_state["last_result"]
    if r.get("is_smalltalk"):
        st.info(r["response"])
    else:
        st.markdown("## 🧾 Pipeline Output")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Detected Language", r["language_name"], f"conf {r['language_confidence']:.0%}")
        m2.metric("Translation Coverage", f"{r['translation_confidence']:.0%}")
        m3.metric("Disaster Type", r["disaster_type"], f"conf {r['disaster_type_confidence']:.0%}")
        m4.metric("Severity", r["severity"], f"conf {r['severity_confidence']:.0%}")

        st.markdown(f"**Severity Level:** {severity_badge(r['severity'])}", unsafe_allow_html=True)

        tabs = st.tabs(["🌐 Translation", "🧹 Preprocessing", "🏷️ Entities (NER)",
                        "📊 Classification Confidence", "🩺 Emergency Guidance", "💬 Assistant Response"])

        with tabs[0]:
            cA, cB = st.columns(2)
            cA.text_area("Original Message", r.get("translated_text_original", user_text), height=120, disabled=True)
            cB.text_area("Translated to English", r["translated_text"], height=120, disabled=True)
            st.caption(f"Lexicon coverage-based translation confidence: **{r['translation_confidence']:.1%}** "
                       "(fraction of characters matched against curated multilingual emergency lexicon)")

        with tabs[1]:
            steps = r["preprocessing"]
            st.write("**Cleaned text:**", steps["cleaned"])
            st.write("**Tokens:**", steps["tokens"])
            st.write("**After stopword removal:**", steps["no_stopwords"])
            st.write("**Lemmatized tokens:**", steps["lemmatized"])
            st.write("**Final feature text fed to classifiers:**")
            st.code(steps["final_text"])

        with tabs[2]:
            ents = r["entities"]
            rows = []
            for label, values in ents.items():
                for v in values:
                    rows.append({"Entity Type": label, "Value": v})
            if rows:
                edf = pd.DataFrame(rows)
                st.dataframe(edf, use_container_width=True, hide_index=True)
                fig = px.bar(edf["Entity Type"].value_counts().reset_index(),
                             x="index" if "index" in edf.columns else "Entity Type",
                             y="Entity Type" if "index" in edf.columns else "count",
                             title="Entity Type Counts")
            else:
                st.warning("No named entities detected in this message.")

        with tabs[3]:
            cc1, cc2 = st.columns(2)
            with cc1:
                dist = r["disaster_type_distribution"]
                ddf = pd.DataFrame(list(dist.items()), columns=["Disaster Type", "Probability"])
                fig = px.bar(ddf, x="Probability", y="Disaster Type", orientation="h",
                             title="Disaster Type — Model Confidence", color="Probability",
                             color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)
            with cc2:
                sdist = r["severity_distribution"]
                sdf = pd.DataFrame(list(sdist.items()), columns=["Severity", "Probability"])
                fig2 = px.bar(sdf, x="Probability", y="Severity", orientation="h",
                              title="Severity — Model Confidence", color="Probability",
                              color_continuous_scale="Oranges")
                st.plotly_chart(fig2, use_container_width=True)

        with tabs[4]:
            for g in r["guidance"]:
                st.markdown(f"<div class='guidance-box'><b>{g['disaster_type']} — {g['question']}</b> "
                            f"<span class='pill'>match {g['score']:.2f}</span><br>{g['answer']}</div>",
                            unsafe_allow_html=True)

        with tabs[5]:
            st.markdown(r["response"])

else:
    st.caption("Enter a message above (or pick a sample) and click **Analyze Emergency Message**.")
