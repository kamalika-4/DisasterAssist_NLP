"""
Named Entity Recognition module.

Uses custom rule-based extraction so the application can run even when
spaCy native extensions are blocked by Windows Application Control.

Extracts:
- Disaster terms
- Casualty / impact counts
- Basic locations
- Basic dates
- Basic organizations
"""

import re
import pandas as pd


# ---------------------------------------------------------
# Disaster keywords
# ---------------------------------------------------------

DISASTER_KEYWORDS = [
    "flood",
    "flooding",
    "earthquake",
    "fire",
    "cyclone",
    "landslide",
    "accident",
    "collapse",
    "collapsed",
    "explosion",
    "tsunami",
    "storm",
]


# ---------------------------------------------------------
# Casualty / impact pattern
# ---------------------------------------------------------

CASUALTY_PATTERN = re.compile(
    r"\b(\d+)\s*(people|persons|injured|families|houses|shops|vehicles)\b",
    re.IGNORECASE
)


# ---------------------------------------------------------
# Basic date pattern
# ---------------------------------------------------------

DATE_PATTERN = re.compile(
    r"\b("
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|"
    r"\d{1,2}\s+"
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
    r"(?:\s+\d{4})?"
    r")\b",
    re.IGNORECASE
)


# ---------------------------------------------------------
# Basic location detection
# ---------------------------------------------------------

KNOWN_LOCATIONS = [
    "Chennai",
    "Coimbatore",
    "Madurai",
    "Bengaluru",
    "Bangalore",
    "Delhi",
    "Mumbai",
    "Kolkata",
    "Hyderabad",
    "Pune",
    "Kerala",
    "Tamil Nadu",
    "India",
]


# ---------------------------------------------------------
# Basic organization detection
# ---------------------------------------------------------

KNOWN_ORGANIZATIONS = [
    "police",
    "fire department",
    "fire service",
    "government",
    "red cross",
    "hospital",
    "ambulance",
    "NDRF",
]


def extract_entities(text: str):
    """
    Extract emergency-related entities.

    This function does not require spaCy, so it works even when
    spaCy's native extensions are blocked.
    """

    entities = {
        "PERSON": [],
        "LOCATION": [],
        "DATE": [],
        "ORG": [],
    }

    # -----------------------------------------------------
    # 1. Disaster terms
    # -----------------------------------------------------

    text_lower = text.lower()

    disaster_terms = sorted(
        set(
            keyword
            for keyword in DISASTER_KEYWORDS
            if keyword in text_lower
        )
    )

    # -----------------------------------------------------
    # 2. Casualty / impact counts
    # -----------------------------------------------------

    casualty_matches = CASUALTY_PATTERN.findall(text)

    casualties = [
        f"{number} {label}"
        for number, label in casualty_matches
    ]

    # -----------------------------------------------------
    # 3. Location extraction
    # -----------------------------------------------------

    for location in KNOWN_LOCATIONS:
        if re.search(
            r"\b" + re.escape(location) + r"\b",
            text,
            re.IGNORECASE
        ):
            entities["LOCATION"].append(location)

    # -----------------------------------------------------
    # 4. Date extraction
    # -----------------------------------------------------

    date_matches = DATE_PATTERN.findall(text)

    entities["DATE"].extend(date_matches)

    # -----------------------------------------------------
    # 5. Organization extraction
    # -----------------------------------------------------

    for organization in KNOWN_ORGANIZATIONS:
        if re.search(
            r"\b" + re.escape(organization) + r"\b",
            text,
            re.IGNORECASE
        ):
            entities["ORG"].append(organization)

    # -----------------------------------------------------
    # 6. Remove duplicates
    # -----------------------------------------------------

    for key in entities:
        entities[key] = sorted(set(entities[key]))

    entities["DISASTER_TERMS"] = disaster_terms
    entities["CASUALTY_COUNTS"] = casualties

    # Return text itself as second value so existing code
    # that expects (entities, doc) does not break.
    return entities, text


def entities_to_dataframe(entities: dict):
    rows = []

    for label, values in entities.items():
        for value in values:
            rows.append({
                "Entity Type": label,
                "Value": value
            })

    if not rows:
        return pd.DataFrame(
            columns=["Entity Type", "Value"]
        )

    return pd.DataFrame(rows)