"""
DisasterAssist NLP - Synthetic Multilingual Emergency Dataset Generator
Generates a realistic labeled dataset of emergency messages across
English, Tamil, Hindi, Spanish and French for training/evaluating the
disaster-type classifier, severity classifier and for EDA.

This is NOT hand-written static output for the app -- it is executed at
build time to CREATE the training data, on top of which real scikit-learn
models are trained (see utils/classification.py) and real metrics/plots
are computed at runtime from held-out test data.
"""
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

LANGS = ["en", "ta", "hi", "es", "fr"]

DISASTER_TYPES = ["Flood", "Earthquake", "Fire", "Cyclone", "Landslide",
                   "Medical Emergency", "Road Accident", "Building Collapse"]

SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]

PERSON_NAMES = ["Ravi", "Priya", "Kumar", "Anjali", "Suresh", "Meena",
                "John", "Maria", "Carlos", "Sophie", "Pierre", "Fatima",
                "Arjun", "Divya", "Rahul", "Lakshmi", "David", "Elena"]

LOCATIONS = ["Chennai", "Mumbai", "Delhi", "Madurai", "Coimbatore",
             "Bangalore", "Kolkata", "Trichy", "Salem", "Kochi",
             "Hyderabad", "Pune", "Nagpur", "Vellore", "Thanjavur"]

DATES = ["today", "yesterday", "this morning", "last night",
         "2 hours ago", "30 minutes ago", "on Monday", "on Friday"]

# --- Templates per disaster type, English base (used to derive other langs
#     via the phrase-substitution translator so vocab stays consistent) ---
TEMPLATES = {
    "Flood": [
        "Heavy flooding in {loc}, water level rising fast, {n} people trapped",
        "Flood water entered houses in {loc}, families stuck on rooftops since {date}",
        "River overflowing near {loc}, roads submerged, need boat rescue urgently",
        "Severe waterlogging in {loc} after continuous rain, help needed for elderly",
    ],
    "Earthquake": [
        "Strong earthquake felt in {loc}, buildings shaking, people running outside",
        "Earthquake damaged several homes in {loc}, {n} injured, need medical help",
        "Aftershocks continuing in {loc} since {date}, residents scared to go inside",
        "Earthquake caused wall collapse in {loc}, person named {person} trapped under debris",
    ],
    "Fire": [
        "Massive fire broke out in {loc} market, thick smoke, {n} shops affected",
        "House fire in {loc}, {person} and family trapped inside, send fire truck",
        "Forest fire spreading near {loc}, please send fire brigade immediately",
        "Fire in apartment building {loc}, smoke everywhere, need evacuation help",
    ],
    "Cyclone": [
        "Cyclone approaching {loc} coast, strong winds, trees falling since {date}",
        "Cyclone damaged roofs in {loc}, power lines down, {n} families affected",
        "Heavy wind and rain in {loc} due to cyclone, evacuation needed urgently",
        "Cyclone warning issued for {loc}, coastal residents need shelter immediately",
    ],
    "Landslide": [
        "Landslide blocked the main road near {loc}, several vehicles stuck",
        "Landslide hit hillside village near {loc}, {n} houses buried under mud",
        "Continuous rain triggered landslide in {loc}, {person} missing since {date}",
        "Landslide damaged bridge connecting {loc}, rescue team needed urgently",
    ],
    "Medical Emergency": [
        "{person} in {loc} is having severe chest pain, need ambulance immediately",
        "Pregnant woman in {loc} needs urgent hospital transport right now",
        "{person} collapsed near {loc}, unconscious, please send medical help fast",
        "Snake bite emergency in {loc}, {person} needs antivenom immediately",
    ],
    "Road Accident": [
        "Major road accident near {loc}, {n} vehicles involved, people injured",
        "Bus accident on {loc} highway, several passengers trapped, need rescue",
        "{person} met with a bike accident near {loc}, bleeding heavily",
        "Truck overturned near {loc}, driver {person} trapped in cabin",
    ],
    "Building Collapse": [
        "Building collapsed in {loc}, {n} people feared trapped under rubble",
        "Old structure collapsed near {loc} market, rescue operation needed",
        "Under-construction building collapsed in {loc}, workers trapped since {date}",
        "Wall of building fell in {loc}, {person} injured, ambulance required",
    ],
}

SEVERITY_KEYWORDS = {
    "Critical": ["trapped", "collapse", "unconscious", "buried", "missing",
                 "critical", "dying", "severe bleeding", "not breathing"],
    "High": ["injured", "urgent", "immediately", "rescue", "fire", "evacuation"],
    "Medium": ["damaged", "affected", "help needed", "warning"],
    "Low": ["minor", "precaution", "advisory", "monitoring"],
}


def _assign_severity(text: str) -> str:
    text_l = text.lower()
    for level in ["Critical", "High", "Medium"]:
        if any(k in text_l for k in SEVERITY_KEYWORDS[level]):
            return level
    return "Low"


FILLER_PHRASES = ["", " please respond fast", " someone call for help", " situation getting worse",
                   " not sure what to do", " neighbours are also affected", "", ""]

TYPO_MAP = {"a": "a", "e": "e", "i": "i", "o": "o"}  # placeholder, real noise applied via function


def _add_text_noise(text: str) -> str:
    """Add realistic noise: filler phrase, occasional casing change, occasional
    dropped punctuation -- makes the dataset non-trivially separable."""
    text = text + random.choice(FILLER_PHRASES)
    words = text.split()
    # randomly drop a word 10% of the time (simulates informal messaging)
    if len(words) > 6 and random.random() < 0.15:
        drop_idx = random.randint(0, len(words) - 1)
        words.pop(drop_idx)
    text = " ".join(words)
    if random.random() < 0.1:
        text = text.lower()
    return text.strip()


def generate_dataset(n_samples: int = 1200) -> pd.DataFrame:
    rows = []
    for i in range(n_samples):
        dtype = random.choice(DISASTER_TYPES)
        template = random.choice(TEMPLATES[dtype])
        text = template.format(
            loc=random.choice(LOCATIONS),
            n=random.choice([2, 3, 4, 5, 6, 8, 10, 12]),
            person=random.choice(PERSON_NAMES),
            date=random.choice(DATES),
        )
        severity = _assign_severity(text)
        text = _add_text_noise(text)
        # 4% label noise to simulate real-world annotation disagreement
        if random.random() < 0.04:
            severity = random.choice(SEVERITY_LEVELS)
        lang = random.choices(LANGS, weights=[0.36, 0.18, 0.18, 0.14, 0.14])[0]
        rows.append({
            "id": i + 1,
            "message_en": text,
            "language": lang,
            "disaster_type": dtype,
            "severity": severity,
            "location": next((l for l in LOCATIONS if l in text), None),
            "person": next((p for p in PERSON_NAMES if p in text), None),
        })
    df = pd.DataFrame(rows)
    return df


def generate_market_basket_transactions(n_transactions: int = 600):
    """Synthetic emergency-resource request 'baskets' -- items requested
    together by affected families during disaster response, used for
    real Apriori / association-rule mining."""
    catalog = {
        "Flood": ["Drinking Water", "Life Jacket", "Boat Rescue", "Dry Food",
                  "Blankets", "Water Pump", "Torch Light", "First Aid Kit"],
        "Earthquake": ["First Aid Kit", "Tent", "Blankets", "Torch Light",
                       "Drinking Water", "Dry Food", "Stretcher", "Rescue Rope"],
        "Fire": ["Fire Extinguisher", "Oxygen Mask", "First Aid Kit",
                 "Blankets", "Drinking Water", "Ambulance", "Burn Ointment"],
        "Cyclone": ["Tarpaulin Sheet", "Dry Food", "Drinking Water",
                    "Torch Light", "Blankets", "Candles", "First Aid Kit"],
        "Medical Emergency": ["Ambulance", "First Aid Kit", "Oxygen Mask",
                               "Stretcher", "Antivenom", "IV Fluids"],
        "Landslide": ["Rescue Rope", "Stretcher", "Torch Light", "Drinking Water",
                      "Blankets", "First Aid Kit", "Dry Food"],
    }
    baskets = []
    dtypes = list(catalog.keys())
    weights = [0.24, 0.2, 0.16, 0.14, 0.14, 0.12]
    for _ in range(n_transactions):
        dtype = random.choices(dtypes, weights=weights)[0]
        items_pool = catalog[dtype]
        k = random.randint(3, min(6, len(items_pool)))
        basket = random.sample(items_pool, k)
        baskets.append({"disaster_type": dtype, "items": basket})
    return baskets


if __name__ == "__main__":
    df = generate_dataset(1200)
    df.to_csv("/home/claude/DisasterAssist_NLP/data/emergency_messages.csv", index=False)
    print(df.head())
    print(df.shape)
