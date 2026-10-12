import streamlit as st

PRIMARY = "#D64545"
DARK = "#1F2937"
ACCENT = "#2E86AB"

def inject_css():
    st.markdown(f"""
    <style>
    .stApp {{ background-color: #0E1117; }}
    .metric-card {{
        background: linear-gradient(135deg, #1a1f2e 0%, #232a3b 100%);
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }}
    .big-title {{
        font-size: 2.1rem; font-weight: 800; color: white;
        margin-bottom: 0px;
    }}
    .subtitle {{ color: #9aa4b2; font-size: 1.0rem; margin-top: 0px;}}
    .severity-critical {{ color: #ff4d4d; font-weight: 700; }}
    .severity-high {{ color: #ff9f45; font-weight: 700; }}
    .severity-medium {{ color: #f4d03f; font-weight: 700; }}
    .severity-low {{ color: #58d68d; font-weight: 700; }}
    .guidance-box {{
        background: #161b28; border-left: 4px solid {PRIMARY};
        padding: 14px 18px; border-radius: 8px; margin-bottom: 10px;
    }}
    .pill {{
        display:inline-block; padding: 3px 12px; border-radius: 999px;
        background:#232a3b; color:#c9d1e0; font-size: 0.8rem; margin-right:6px;
    }}
    </style>
    """, unsafe_allow_html=True)


def page_header(title, subtitle=""):
    st.markdown(f"<div class='big-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='subtitle'>{subtitle}</div>", unsafe_allow_html=True)
    st.markdown("---")


def severity_badge(sev: str) -> str:
    cls = {"Critical": "severity-critical", "High": "severity-high",
           "Medium": "severity-medium", "Low": "severity-low"}.get(sev, "")
    return f"<span class='{cls}'>{sev}</span>"
