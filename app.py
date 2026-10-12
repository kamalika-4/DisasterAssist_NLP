import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.app_config import inject_css, page_header
from utils.data_generator import generate_dataset
from utils.classification import train_all_models
from utils.qa_engine import build_qa_index

st.set_page_config(page_title="DisasterAssist NLP", page_icon="🚨", layout="wide")
inject_css()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "emergency_messages.csv")


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        df = generate_dataset(1200)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource(show_spinner="Training NLP/ML models (first run only)...")
def init_models():
    metrics = train_all_models()
    build_qa_index()
    return metrics


df = load_data()
metrics = init_models()

page_header("🚨 DisasterAssist NLP",
            "Multilingual Emergency Message Understanding & Response System &nbsp;|&nbsp; "
            "Translation · NER · Classification · Speech · QA · Chatbot · Market Basket Analysis")

# ---------------- KPI Row ----------------
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown("<div class='metric-card'><h2>%d</h2>Messages in Corpus</div>" % len(df), unsafe_allow_html=True)
with c2:
    st.markdown("<div class='metric-card'><h2>5</h2>Languages Supported</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='metric-card'><h2>%d</h2>Disaster Types</div>" % df["disaster_type"].nunique(), unsafe_allow_html=True)
with c4:
    acc = metrics["disaster_type"]["accuracy"] * 100
    st.markdown(f"<div class='metric-card'><h2>{acc:.1f}%</h2>Disaster-Type Model Accuracy</div>", unsafe_allow_html=True)
with c5:
    acc2 = metrics["severity"]["accuracy"] * 100
    st.markdown(f"<div class='metric-card'><h2>{acc2:.1f}%</h2>Severity Model Accuracy</div>", unsafe_allow_html=True)

st.write("")


st.markdown("---")
st.markdown("## 🔄 ETL Pipeline (Extract → Transform → Load)")
e1, e2, e3 = st.columns(3)
with e1:
    st.markdown("#### 1️⃣ Extract")
    st.write("Raw multilingual emergency messages ingested from user input / synthetic corpus generation "
              "simulating real citizen reports across 5 languages.")
    st.dataframe(df[["id", "language", "message_en"]].head(5), use_container_width=True, hide_index=True)
with e2:
    st.markdown("#### 2️⃣ Transform")
    st.write("Cleaning → tokenization → stopword removal → lemmatization → TF-IDF vectorization; "
             "labels derived for disaster type & severity.")
    sample_txt = df["message_en"].iloc[0]
    from utils.preprocessing import preprocess_pipeline
    steps = preprocess_pipeline(sample_txt)
    st.code(f"Original : {steps['original']}\n"
            f"Cleaned  : {steps['cleaned']}\n"
            f"Tokens   : {steps['tokens'][:8]}...\n"
            f"Lemmas   : {steps['lemmatized'][:8]}...", language="text")
with e3:
    st.markdown("#### 3️⃣ Load")
    st.write("Processed features loaded into scikit-learn pipelines, trained & cached models saved "
             "with joblib for low-latency inference across all app pages.")
    st.dataframe(df[["disaster_type", "severity"]].value_counts().reset_index(name="count").head(6),
                 use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("## 📊 Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)
with col1:
    lang_counts = df["language"].value_counts().reset_index()
    lang_counts.columns = ["Language", "Count"]
    lang_map = {"en": "English", "ta": "Tamil", "hi": "Hindi", "es": "Spanish", "fr": "French"}
    lang_counts["Language"] = lang_counts["Language"].map(lang_map)
    fig = px.pie(lang_counts, names="Language", values="Count", hole=0.45,
                 title="Message Distribution by Language",
                 color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    dtype_counts = df["disaster_type"].value_counts().reset_index()
    dtype_counts.columns = ["Disaster Type", "Count"]
    fig2 = px.bar(dtype_counts, x="Disaster Type", y="Count", title="Disaster Type Frequency",
                  color="Count", color_continuous_scale="Reds")
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    sev_counts = df["severity"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).reset_index()
    sev_counts.columns = ["Severity", "Count"]
    fig3 = px.bar(sev_counts, x="Severity", y="Count", title="Severity Level Distribution",
                  color="Severity",
                  color_discrete_map={"Low": "#58d68d", "Medium": "#f4d03f",
                                       "High": "#ff9f45", "Critical": "#ff4d4d"})
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    df["msg_len"] = df["message_en"].str.split().apply(len)
    fig4 = px.histogram(df, x="msg_len", nbins=25, title="Message Length Distribution (words)",
                         color_discrete_sequence=["#2E86AB"])
    fig4.update_layout(xaxis_title="Word Count", yaxis_title="Frequency")
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("#### Cross-tab: Disaster Type vs Severity")
cross = pd.crosstab(df["disaster_type"], df["severity"])
cross = cross[[c for c in ["Low", "Medium", "High", "Critical"] if c in cross.columns]]
fig5 = px.imshow(cross, text_auto=True, aspect="auto", color_continuous_scale="Reds",
                  title="Heatmap: Disaster Type vs Severity Level")
st.plotly_chart(fig5, use_container_width=True)

with st.expander("🔍 View raw dataset sample"):
    st.dataframe(df.drop(columns=["msg_len"], errors="ignore").sample(15, random_state=1),
                 use_container_width=True, hide_index=True)

st.markdown("---")
st.info("👉 Use the **sidebar navigation** to explore Live Emergency Assistant, NLP Analysis, "
        "Speech Processing, Analytics, Emergency Chatbot, and Market Basket Analysis pages.")
