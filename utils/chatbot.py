"""
Contextual Emergency Chatbot engine.
Combines: language detection -> translation -> preprocessing ->
disaster-type classification -> severity classification -> NER ->
QA retrieval, then composes a contextual natural-language response.
Maintains simple conversational state (last known disaster context)
in the caller's session for follow-up turns.
"""
from utils.preprocessing import detect_language, preprocess_pipeline
from utils.translation import translate_to_english, LANG_NAMES
from utils.classification import predict_disaster_type, predict_severity
from utils.ner_utils import extract_entities
from utils.qa_engine import answer_query

GREETING_WORDS = {"hi", "hello", "hey", "good morning", "good evening", "namaste", "vanakkam"}
THANKS_WORDS = {"thanks", "thank you", "thank", "ok thanks", "great thanks"}

SEVERITY_TONE = {
    "Critical": "🔴 This sounds CRITICAL. Prioritizing your message for immediate dispatch.",
    "High": "🟠 This looks like a HIGH severity situation. Emergency guidance below.",
    "Medium": "🟡 This looks MODERATE. Please follow the guidance and stay alert.",
    "Low": "🟢 This looks LOW severity, but here is relevant safety guidance.",
}


def process_emergency_message(raw_text: str, session_state: dict = None):
    """Full pipeline used by both the Live Assistant and Chatbot pages."""
    session_state = session_state if session_state is not None else {}
    result = {}

    text_l = raw_text.strip().lower()
    if text_l in GREETING_WORDS:
        result["is_smalltalk"] = True
        result["response"] = ("Hello, I am DisasterAssist. Please describe your emergency "
                               "in English, Tamil, Hindi, Spanish, or French and I will "
                               "detect the language, understand the situation, and provide guidance.")
        return result
    if any(t in text_l for t in THANKS_WORDS) and len(text_l.split()) <= 4:
        result["is_smalltalk"] = True
        result["response"] = "You're welcome. Stay safe. Send another message if you need more help."
        return result

    # 1. Language detection
    lang, lang_conf = detect_language(raw_text)
    result["language"] = lang
    result["language_name"] = LANG_NAMES.get(lang, lang)
    result["language_confidence"] = lang_conf

    # 2. Translation to English (if needed)
    translated_text, translation_conf = translate_to_english(raw_text, lang)
    result["translated_text"] = translated_text
    result["translation_confidence"] = translation_conf

    # 3. Preprocessing
    steps = preprocess_pipeline(translated_text)
    result["preprocessing"] = steps

    # 4. Disaster type + severity classification (on translated English text)
    dtype, dtype_conf, dtype_dist = predict_disaster_type(translated_text)
    sev, sev_conf, sev_dist = predict_severity(translated_text)
    result["disaster_type"] = dtype
    result["disaster_type_confidence"] = dtype_conf
    result["disaster_type_distribution"] = dtype_dist
    result["severity"] = sev
    result["severity_confidence"] = sev_conf
    result["severity_distribution"] = sev_dist

    # 5. NER
    entities, _ = extract_entities(translated_text)
    result["entities"] = entities

    # 6. QA / Guidance retrieval
    guidance = answer_query(translated_text, disaster_type=dtype, top_k=3)
    result["guidance"] = guidance

    # 7. Update session context
    session_state["last_disaster_type"] = dtype
    session_state["last_severity"] = sev
    if entities.get("LOCATION"):
        session_state["last_location"] = entities["LOCATION"][0]
    elif "last_location" not in session_state:
        session_state["last_location"] = None

    # 8. Compose contextual chatbot response
    tone = SEVERITY_TONE.get(sev, "")
    loc_txt = f" in {entities['LOCATION'][0]}" if entities["LOCATION"] else ""
    top_guidance = guidance[0]["answer"] if guidance else "Please contact your local emergency helpline immediately."
    response_lines = [
        f"I detected your message in **{result['language_name']}** and identified this as a "
        f"**{dtype}** emergency{loc_txt} with **{sev}** severity "
        f"(type confidence {dtype_conf:.0%}, severity confidence {sev_conf:.0%}).",
        tone,
        f"**Recommended action:** {top_guidance}",
    ]
    if entities["CASUALTY_COUNTS"]:
        response_lines.append(f"⚠️ Reported impact: {', '.join(entities['CASUALTY_COUNTS'])}.")
    if entities["PERSON"]:
        response_lines.append(f"👤 Person(s) mentioned: {', '.join(entities['PERSON'])}.")

    result["response"] = "\n\n".join(response_lines)
    result["is_smalltalk"] = False
    return result
