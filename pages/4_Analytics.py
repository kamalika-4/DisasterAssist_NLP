import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.app_config import inject_css, page_header
from utils.classification import train_all_models

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")
inject_css()
page_header("📈 Model Analytics & Evaluation", "Real held-out test metrics computed from trained scikit-learn models")

metrics = train_all_models()

model_choice = st.selectbox("Select model to inspect", ["Disaster Type Classifier (Logistic Regression)",
                                                          "Severity Classifier (Random Forest)"])
key = "disaster_type" if "Disaster Type" in model_choice else "severity"
m = metrics[key]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{m['accuracy']:.2%}")
c2.metric("F1 (macro)", f"{m['f1_macro']:.2%}")
c3.metric("Precision (macro)", f"{m['precision_macro']:.2%}")
c4.metric("Recall (macro)", f"{m['recall_macro']:.2%}")
st.caption(f"Trained on {m['n_train']} messages · evaluated on {m['n_test']} held-out unseen messages "
           "(80/20 stratified split).")

st.markdown("### 🔲 Confusion Matrix")
cm = np.array(m["confusion_matrix"])
labels = m["labels"]
fig = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Reds",
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16}
    )
)

fig.update_layout(
    xaxis_title="Predicted",
    yaxis_title="Actual"
)
st.plotly_chart(fig)
fig.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label", height=500)
fig.update_xaxes(side="bottom")
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 📋 Per-Class Report")
report_df = pd.DataFrame(m["report"]).transpose()
report_df = report_df.round(3)
st.dataframe(report_df, use_container_width=True)

st.markdown("---")
st.markdown("### 📊 Model Comparison Summary")
comp_rows = []
for k, name in [("disaster_type", "Disaster Type (LogReg)"), ("severity", "Severity (RandomForest)")]:
    mm = metrics[k]
    comp_rows.append({"Model": name, "Accuracy": mm["accuracy"], "F1 Macro": mm["f1_macro"],
                       "Precision": mm["precision_macro"], "Recall": mm["recall_macro"]})
comp_df = pd.DataFrame(comp_rows)
fig2 = px.bar(comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
              x="Model", y="Score", color="Metric", barmode="group",
              title="Classifier Performance Comparison", range_y=[0, 1.05])
st.plotly_chart(fig2, use_container_width=True)

with st.expander("ℹ️ Methodology"):
    st.markdown("""
    - **Feature extraction**: TF-IDF (1-2 grams, up to 3000 features, sublinear TF scaling)
    - **Disaster-type model**: Logistic Regression (`C=5.0`, `max_iter=1000`)
    - **Severity model**: Random Forest (300 trees)
    - **Split**: 80% train / 20% test, stratified by label, `random_state=42`
    - **Data**: synthetically generated multilingual emergency corpus with realistic
      text noise (informal filler phrases, dropped words, ~4% label noise) to avoid
      an unrealistically trivial classification task.
    """)
