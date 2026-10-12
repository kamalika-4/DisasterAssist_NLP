import os, sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.app_config import inject_css, page_header
from utils.chatbot import process_emergency_message

st.set_page_config(page_title="Emergency Chatbot", page_icon="💬", layout="wide")
inject_css()
page_header("💬 DisasterAssist Emergency Chatbot", "Multilingual, context-aware conversational assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello, I am **DisasterAssist**. Describe your emergency in "
                                          "English, Tamil, Hindi, Spanish, or French and I'll detect the "
                                          "language, assess severity, and give you guidance."}
    ]
if "chat_context" not in st.session_state:
    st.session_state.chat_context = {}

st.sidebar.markdown("### 🗂️ Conversation Context")
context_placeholder = st.sidebar.empty()

def render_conversation_context():
    ctx = st.session_state.chat_context
    disaster_type = ctx.get("last_disaster_type") or st.session_state.get("last_disaster_type") or "—"
    severity = ctx.get("last_severity") or st.session_state.get("last_severity") or "—"
    location = ctx.get("last_location") or st.session_state.get("last_location") or "—"
    with context_placeholder.container():
        st.write(f"**Last disaster type:** {disaster_type}")
        st.write(f"**Last severity:** {severity}")
        st.write(f"**Last location:** {location}")

render_conversation_context()

if st.sidebar.button("🔄 Reset conversation"):
    st.session_state.chat_history = [{"role": "assistant", "content": "Conversation reset. How can I help?"}]
    st.session_state.chat_context = {}
    st.session_state["last_disaster_type"] = None
    st.session_state["last_severity"] = None
    st.session_state["last_location"] = None
    st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_msg = st.chat_input("Type your emergency message here (any of 5 supported languages)...")

if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            result = process_emergency_message(user_msg, st.session_state.chat_context)
        st.markdown(result["response"])
        if not result.get("is_smalltalk"):
            st.session_state.chat_context["last_disaster_type"] = result.get("disaster_type")
            st.session_state.chat_context["last_severity"] = result.get("severity")
            ents = result.get("entities", {})
            locs = ents.get("LOCATION", [])
            if locs:
                st.session_state.chat_context["last_location"] = locs[0]
            st.session_state["last_disaster_type"] = st.session_state.chat_context.get("last_disaster_type")
            st.session_state["last_severity"] = st.session_state.chat_context.get("last_severity")
            st.session_state["last_location"] = st.session_state.chat_context.get("last_location")

            with st.expander("🔍 See detected details"):
                st.write(f"- Language: **{result['language_name']}** ({result['language_confidence']:.0%} confidence)")
                st.write(f"- Translated text: {result['translated_text']}")
                st.write(f"- Disaster type: **{result['disaster_type']}** ({result['disaster_type_confidence']:.0%})")
                st.write(f"- Severity: **{result['severity']}** ({result['severity_confidence']:.0%})")
                if any(ents.values()):
                    st.write("- Entities:", {k: v for k, v in ents.items() if v})
    st.session_state.chat_history.append({"role": "assistant", "content": result["response"]})
    render_conversation_context()

st.markdown("---")
st.caption("💡 Try: *'water is rising fast near Madurai, elderly couple trapped'* or "
           "*'मुंबई में आग लगी है, तुरंत मदद चाहिए'* or *'accident sur la route près de Chennai'*")
