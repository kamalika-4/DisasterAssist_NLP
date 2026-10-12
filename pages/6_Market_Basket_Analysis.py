import os, sys
import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.app_config import inject_css, page_header
from utils.market_basket import run_market_basket_analysis, item_frequency_table

st.set_page_config(page_title="Market Basket Analysis", page_icon="🛒", layout="wide")
inject_css()
page_header("🛒 Emergency Relief — Market Basket Analysis", "Apriori association-rule mining over relief-resource requests")

st.markdown("Discovering which **relief items are requested together** during disasters helps "
            "response teams pre-pack and dispatch combined relief kits more efficiently.")

c1, c2, c3 = st.columns(3)
n_tx = c1.slider("Number of transactions", 200, 2000, 600, step=100)
min_sup = c2.slider("Minimum support", 0.02, 0.3, 0.08, step=0.01)
min_conf = c3.slider("Minimum confidence", 0.1, 0.9, 0.4, step=0.05)

baskets, freq_itemsets, rules = run_market_basket_analysis(min_support=min_sup, min_confidence=min_conf, n_transactions=n_tx)

st.markdown(f"Generated **{len(baskets)}** synthetic relief-request transactions across 6 disaster types.")

tab1, tab2, tab3, tab4 = st.tabs(["📦 Item Frequency", "🔗 Frequent Itemsets", "📐 Association Rules", "🕸️ Rule Explorer"])

with tab1:
    freq_df = item_frequency_table(baskets)
    fig = px.bar(freq_df, x="Item", y="Frequency", title="Relief Item Request Frequency",
                 color="Frequency", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(freq_df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown(f"**{len(freq_itemsets)}** frequent itemsets found (support ≥ {min_sup}).")
    disp = freq_itemsets[["itemsets_str", "support"]].rename(
        columns={"itemsets_str": "Itemset", "support": "Support"})
    st.dataframe(disp, use_container_width=True, hide_index=True)
    top = disp.sort_values("Support", ascending=False).head(15)
    fig2 = px.bar(top, x="Support", y="Itemset", orientation="h", title="Top Frequent Itemsets")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    if rules.empty:
        st.warning("No association rules found at this support/confidence threshold. Try lowering them.")
    else:
        st.markdown(f"**{len(rules)}** association rules discovered (confidence ≥ {min_conf}).")
        disp_rules = rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].rename(
            columns={"antecedents_str": "If items requested", "consequents_str": "Then also request",
                     "support": "Support", "confidence": "Confidence", "lift": "Lift"})
        st.dataframe(disp_rules.round(3), use_container_width=True, hide_index=True)

        fig3 = px.scatter(rules, x="support", y="confidence", size="lift", color="lift",
                           hover_data=["antecedents_str", "consequents_str"],
                           title="Support vs Confidence (bubble size/color = Lift)",
                           color_continuous_scale="Turbo")
        st.plotly_chart(fig3, use_container_width=True)

with tab4:
    if rules.empty:
        st.info("No rules to explore — adjust thresholds in the sliders above.")
    else:
        st.markdown("### 🔍 Get recommended items for a relief kit")
        all_items = sorted(set(item for b in baskets for item in b["items"]))
        chosen = st.multiselect("Select item(s) already being dispatched", all_items, default=[all_items[0]])
        if chosen:
            chosen_set = set(chosen)
            matched = rules[rules["antecedents"].apply(lambda a: chosen_set.issubset(set(a)))]
            matched = matched.sort_values("lift", ascending=False)
            if matched.empty:
                st.info("No strong association rules found for this exact combination. Try a single item.")
            else:
                st.success(f"Found {len(matched)} matching rule(s):")
                for _, row in matched.head(8).iterrows():
                    st.markdown(f"<div class='guidance-box'>If dispatching <b>{', '.join(chosen)}</b>, "
                                f"also consider: <b>{row['consequents_str']}</b> "
                                f"<span class='pill'>confidence {row['confidence']:.0%}</span>"
                                f"<span class='pill'>lift {row['lift']:.2f}</span></div>",
                                unsafe_allow_html=True)
