"""
Disaster-type and severity text classifiers.
Real scikit-learn pipeline: TF-IDF vectorizer + LogisticRegression /
RandomForest, trained on a held-out split of the generated dataset, with
genuine accuracy/F1/confusion-matrix computed on the unseen test split.
Models are cached to disk with joblib so the Streamlit app trains once.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "emergency_messages.csv")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_all_models(force: bool = False):
    """Train disaster-type and severity classifiers; return metrics dict."""
    dtype_path = os.path.join(MODEL_DIR, "disaster_type_model.joblib")
    sev_path = os.path.join(MODEL_DIR, "severity_model.joblib")
    metrics_path = os.path.join(MODEL_DIR, "metrics.joblib")

    if not force and os.path.exists(dtype_path) and os.path.exists(sev_path) and os.path.exists(metrics_path):
        return joblib.load(metrics_path)

    df = pd.read_csv(DATA_PATH)
    X = df["message_en"].astype(str)

    metrics = {}

    # ---- Disaster type classifier ----
    y_type = df["disaster_type"]
    Xtr, Xte, ytr, yte = train_test_split(X, y_type, test_size=0.2, random_state=42, stratify=y_type)
    vec_type = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
    Xtr_v = vec_type.fit_transform(Xtr)
    Xte_v = vec_type.transform(Xte)
    clf_type = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
    clf_type.fit(Xtr_v, ytr)
    pred_type = clf_type.predict(Xte_v)

    labels_type = sorted(y_type.unique())
    metrics["disaster_type"] = {
        "accuracy": accuracy_score(yte, pred_type),
        "f1_macro": f1_score(yte, pred_type, average="macro"),
        "precision_macro": precision_score(yte, pred_type, average="macro", zero_division=0),
        "recall_macro": recall_score(yte, pred_type, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(yte, pred_type, labels=labels_type).tolist(),
        "labels": labels_type,
        "report": classification_report(yte, pred_type, output_dict=True, zero_division=0),
        "n_train": len(Xtr), "n_test": len(Xte),
    }
    joblib.dump({"vectorizer": vec_type, "model": clf_type}, dtype_path)

    # ---- Severity classifier ----
    y_sev = df["severity"]
    Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y_sev, test_size=0.2, random_state=42, stratify=y_sev)
    vec_sev = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
    Xtr2_v = vec_sev.fit_transform(Xtr2)
    Xte2_v = vec_sev.transform(Xte2)
    clf_sev = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    clf_sev.fit(Xtr2_v, ytr2)
    pred_sev = clf_sev.predict(Xte2_v)

    labels_sev = ["Low", "Medium", "High", "Critical"]
    metrics["severity"] = {
        "accuracy": accuracy_score(yte2, pred_sev),
        "f1_macro": f1_score(yte2, pred_sev, average="macro"),
        "precision_macro": precision_score(yte2, pred_sev, average="macro", zero_division=0),
        "recall_macro": recall_score(yte2, pred_sev, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(yte2, pred_sev, labels=labels_sev).tolist(),
        "labels": labels_sev,
        "report": classification_report(yte2, pred_sev, output_dict=True, zero_division=0),
        "n_train": len(Xtr2), "n_test": len(Xte2),
    }
    joblib.dump({"vectorizer": vec_sev, "model": clf_sev}, sev_path)

    joblib.dump(metrics, metrics_path)
    return metrics


def load_models():
    dtype_path = os.path.join(MODEL_DIR, "disaster_type_model.joblib")
    sev_path = os.path.join(MODEL_DIR, "severity_model.joblib")
    return joblib.load(dtype_path), joblib.load(sev_path)


def predict_disaster_type(text: str):
    bundle, _ = load_models()
    vec, model = bundle["vectorizer"], bundle["model"]
    Xv = vec.transform([text])
    pred = model.predict(Xv)[0]
    proba = model.predict_proba(Xv)[0]
    classes = model.classes_
    conf = float(np.max(proba))
    dist = dict(sorted(zip(classes, proba.tolist()), key=lambda x: -x[1]))
    return pred, conf, dist


def predict_severity(text: str):
    _, bundle = load_models()
    vec, model = bundle["vectorizer"], bundle["model"]
    Xv = vec.transform([text])
    pred = model.predict(Xv)[0]
    proba = model.predict_proba(Xv)[0]
    classes = model.classes_
    conf = float(np.max(proba))
    dist = dict(sorted(zip(classes, proba.tolist()), key=lambda x: -x[1]))
    return pred, conf, dist


if __name__ == "__main__":
    m = train_all_models(force=True)
    print("Disaster type acc:", m["disaster_type"]["accuracy"])
    print("Severity acc:", m["severity"]["accuracy"])
