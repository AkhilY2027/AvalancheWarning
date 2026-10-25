"""Predict avalanche likelihood and conditional destructive size. Research prototype only."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from config import MODEL_FILE
from src.features import engineer_features


def category(score: float) -> str:
    if score < .5: return "Low"
    if score < 1.0: return "Moderate"
    if score < 1.5: return "Considerable"
    if score < 2.5: return "High"
    return "Extreme"


def predict(model_path: Path, row: dict) -> dict:
    bundle = joblib.load(model_path)
    features = engineer_features(pd.DataFrame([row]))
    X = features.reindex(columns=bundle["features"])
    probability = float(bundle["occurrence"].predict_proba(X)[0, 1])
    expected_size = float(bundle["size"].predict(X)[0])
    expected_size = min(5.0, max(1.0, expected_size))
    score = probability * expected_size
    return {"notice": "RESEARCH PROTOTYPE — NOT FOR REAL SAFETY DECISIONS",
            "probability": probability, "expected_size_D_given_avalanche": expected_size,
            "risk_score": score, "risk_category": category(score)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=MODEL_FILE)
    p.add_argument("--input", type=Path, required=True, help="CSV with one or more feature rows")
    p.add_argument("--output", type=Path, help="Optional output CSV; otherwise print JSON")
    a = p.parse_args()
    bundle = joblib.load(a.model)
    rows = pd.read_csv(a.input).to_dict(orient="records")
    predictions = [predict(a.model, row) for row in rows]
    result = pd.DataFrame(predictions)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(a.output, index=False)
    else:
        print(result.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
