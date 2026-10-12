import os, sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.app_config import inject_css, page_header
from utils.speech_utils import transcribe_audio_bytes, analyze_audio_signal, synthesize_speech
from utils.chatbot import process_emergency_message

st.set_page_config(page_title="Speech Processing", page_icon="🎙️", layout="wide")
inject_css()
page_header("🎙️ Speech Processing", "Speech-to-Text (Google Speech Recognition) · Signal analysis · Text-to-Speech playback")

tab1, tab2 = st.tabs(["🎤 Voice → Emergency Analysis (STT)", "🔊 Text → Speech (TTS)"])

with tab1:
    st.markdown("### Upload or record an emergency voice message")
    st.caption("Accepts WAV audio. Uses Google Speech Recognition (English). "
               "Signal statistics are computed for any audio.")

    mode = st.radio("Input method", ["🎙️ Record with microphone", "📁 Upload audio file"], horizontal=True)
    audio_bytes = None
    filename_hint = "audio.wav"

    if mode.startswith("🎙️"):
        rec = st.audio_input("Record your emergency message")
        if rec is not None:
            audio_bytes = rec.getvalue()
    else:
        up = st.file_uploader("Upload a WAV file", type=["wav"])
        if up is not None:
            audio_bytes = up.getvalue()
            filename_hint = up.name

    if audio_bytes:
        st.audio(audio_bytes)
        stats = analyze_audio_signal(audio_bytes)
        if "error" not in stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Duration", f"{stats['duration_sec']} s")
            c2.metric("Sample Rate", f"{stats['sample_rate']} Hz")
            c3.metric("Channels", stats["channels"])
            c4.metric("RMS Energy", stats["rms_energy"])
        else:
            st.warning(f"Could not compute signal stats (file may not be standard WAV): {stats['error']}")

        if st.button("📝 Transcribe & Analyze Emergency", type="primary"):
            with st.spinner("Running speech recognition..."):
                res = transcribe_audio_bytes(audio_bytes, filename_hint)
            if res["success"]:
                st.success(f"Transcribed via **{res['engine']}**")
                st.text_area("Transcript", res["text"], height=80)
                with st.spinner("Running NLP pipeline on transcript..."):
                    result = process_emergency_message(res["text"], {})
                if not result.get("is_smalltalk"):
                    st.markdown(f"**Disaster Type:** {result['disaster_type']} "
                                f"({result['disaster_type_confidence']:.0%})  |  "
                                f"**Severity:** {result['severity']} ({result['severity_confidence']:.0%})")
                    st.markdown(result["response"])
                else:
                    st.info(result["response"])
            else:
                st.error(f"Transcription failed: {res['error']}")
                st.caption("Tip: Speech recognition works best with clear English speech. "
                           "Very short or noisy clips may not be recognized.")

with tab2:
    st.markdown("### Convert emergency guidance text into speech")
    lang = st.selectbox("Language", ["en", "ta", "hi", "es", "fr"],
                         format_func=lambda x: {"en": "English", "ta": "Tamil", "hi": "Hindi",
                                                 "es": "Spanish", "fr": "French"}[x])
    tts_text = st.text_area("Text to synthesize",
                             "Move to higher ground immediately. Help is on the way.")
    if st.button("🔊 Generate Speech"):
        with st.spinner("Synthesizing speech..."):
            res = synthesize_speech(tts_text, lang)
        if res["success"]:
            st.audio(res["audio_bytes"], format="audio/mp3")
        else:
            st.warning("Text-to-speech requires outbound internet access to the Google TTS endpoint, "
                       "which is not reachable in this sandboxed environment. "
                       f"(Engine error: {res['error']})")
