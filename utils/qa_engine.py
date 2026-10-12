"""
Retrieval-based Question Answering / Guidance engine.
Builds a TF-IDF index over an emergency-response knowledge base and
retrieves the best-matching guidance document(s) for a query using
cosine similarity -- computed live, not hard-coded per query.
"""
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

KNOWLEDGE_BASE = [
    {"disaster_type": "Flood", "question": "What should I do during a flood?",
     "answer": "Move to higher ground immediately. Avoid walking or driving through "
               "flood water. Turn off electricity at the main switch. Keep emergency "
               "contacts and a torch ready. If trapped, signal rescue teams from the "
               "highest point of the building and call the flood helpline."},
    {"disaster_type": "Flood", "question": "How to rescue someone stuck in flood water?",
     "answer": "Do not enter fast-moving water yourself. Call boat rescue teams, throw "
               "a rope or floating object if safe, and guide the person toward higher "
               "ground. Contact the district disaster control room immediately."},
    {"disaster_type": "Earthquake", "question": "What to do during an earthquake?",
     "answer": "Drop, Cover, and Hold On. Get under a sturdy table, stay away from "
               "windows and glass, and if outdoors move to open ground away from "
               "buildings and power lines. After shaking stops, check for injuries "
               "and gas leaks before re-entering buildings."},
    {"disaster_type": "Earthquake", "question": "Building collapsed after earthquake, what now?",
     "answer": "Do not move heavy debris alone. Call rescue services immediately, "
               "mark the location clearly, listen for sounds of survivors, and provide "
               "first aid only if it is safe to reach the injured person."},
    {"disaster_type": "Fire", "question": "What should I do in case of a fire?",
     "answer": "Alert everyone nearby, call the fire brigade immediately, stay low to "
               "avoid smoke inhalation, use stairs not elevators, and do not re-enter "
               "a burning building for belongings. Cover nose and mouth with a wet cloth."},
    {"disaster_type": "Fire", "question": "How to handle a forest fire emergency?",
     "answer": "Move away from the fire in the direction opposite to wind flow, alert "
               "the forest fire helpline, avoid dry grass paths, and evacuate low-lying "
               "areas that may trap smoke."},
    {"disaster_type": "Cyclone", "question": "How to stay safe during a cyclone?",
     "answer": "Stay indoors away from windows, store drinking water and dry food, "
               "charge phones and torches in advance, and follow evacuation orders from "
               "local authorities for coastal areas immediately."},
    {"disaster_type": "Landslide", "question": "What to do during a landslide?",
     "answer": "Move away from the slide path immediately to stable ground, avoid river "
               "valleys and low-lying areas, watch for cracking sounds or tilting trees "
               "as warning signs, and report blocked roads to rescue authorities."},
    {"disaster_type": "Medical Emergency", "question": "Someone is having chest pain, what should I do?",
     "answer": "Call an ambulance immediately, keep the person calm and seated upright, "
               "loosen tight clothing, and if trained, be ready to perform CPR if they "
               "become unresponsive and stop breathing."},
    {"disaster_type": "Medical Emergency", "question": "What to do for a snake bite?",
     "answer": "Keep the person still and calm, immobilize the bitten limb below heart "
               "level, do not cut or suck the wound, and get to a hospital immediately "
               "for antivenom treatment."},
    {"disaster_type": "Road Accident", "question": "What to do at a road accident scene?",
     "answer": "Call an ambulance and the police immediately, do not move injured people "
               "unless there is fire or further danger, control bleeding with firm "
               "pressure, and keep the area safe from further traffic."},
    {"disaster_type": "Building Collapse", "question": "How to respond to a building collapse?",
     "answer": "Evacuate the surrounding area, call structural rescue teams immediately, "
               "do not attempt to remove debris without equipment, and guide rescue teams "
               "to any sounds of trapped survivors."},
    {"disaster_type": "General", "question": "What is the national emergency helpline number?",
     "answer": "In India dial 112 for all-in-one emergency services (police, fire, "
               "ambulance). Always share your exact location and nature of emergency "
               "clearly with the operator."},
    {"disaster_type": "General", "question": "How to prepare an emergency kit?",
     "answer": "Keep drinking water, dry food, torch, first-aid kit, whistle, important "
               "documents, phone charger/power bank, and a list of emergency contacts "
               "ready in a waterproof bag at all times."},
]

QA_MODEL_PATH = os.path.join(MODEL_DIR, "qa_index.joblib")


def build_qa_index(force: bool = False):
    if not force and os.path.exists(QA_MODEL_PATH):
        return joblib.load(QA_MODEL_PATH)
    corpus = [f"{d['question']} {d['answer']}" for d in KNOWLEDGE_BASE]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    bundle = {"vectorizer": vectorizer, "matrix": matrix, "kb": KNOWLEDGE_BASE}
    joblib.dump(bundle, QA_MODEL_PATH)
    return bundle


def answer_query(query: str, disaster_type: str = None, top_k: int = 3):
    bundle = build_qa_index()
    vectorizer, matrix, kb = bundle["vectorizer"], bundle["matrix"], bundle["kb"]
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()

    # Boost score for entries matching the predicted disaster type
    if disaster_type:
        for i, doc in enumerate(kb):
            if doc["disaster_type"].lower() == disaster_type.lower():
                sims[i] *= 1.35

    ranked_idx = np.argsort(-sims)[:top_k]
    results = []
    for idx in ranked_idx:
        results.append({
            "question": kb[idx]["question"],
            "answer": kb[idx]["answer"],
            "disaster_type": kb[idx]["disaster_type"],
            "score": round(float(sims[idx]), 4),
        })
    return results


if __name__ == "__main__":
    build_qa_index(force=True)
    for r in answer_query("water is rising in my house, help", disaster_type="Flood"):
        print(r["score"], r["question"])
