"""Train a two-stage avalanche occurrence and conditional-size model."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error

from config import MODEL_FILE, REPORT_FILE, TRAINING_FILE
from src.features import engineer_features, predictor_columns


def train(data_path: Path, model_path: Path, report_path: Path):
    raw = pd.read_parquet(data_path)
    if "avalanche" not in raw or raw.avalanche.nunique() < 2:
        raise ValueError("Training data needs both positive observations and sampled non-event rows")
    features = engineer_features(raw)
    columns = predictor_columns(features)
    if not columns:
        raise ValueError("No usable predictor columns")
    X = features[columns]
    y = raw.avalanche.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    occurrence = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=100, random_state=42))
    occurrence.fit(X_train, y_train)
    metrics = {"occurrence_roc_auc": float(roc_auc_score(y_test, occurrence.predict_proba(X_test)[:, 1]))}
    size_mask = raw.avalanche.eq(1) & raw.size_d.notna()
    if size_mask.sum() < 2:
        raise ValueError("Need at least two avalanche events with known D-scale size")
    size_model = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(max_iter=100, random_state=42))
    size_model.fit(features.loc[size_mask, columns], raw.loc[size_mask, "size_d"].astype(float))
    size_hat = size_model.predict(features.loc[size_mask, columns])
    metrics["conditional_size_mae_train"] = float(mean_absolute_error(raw.loc[size_mask, "size_d"], size_hat))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"occurrence": occurrence, "size": size_model, "features": columns,
                 "research_prototype": True, "version": 1}, model_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=TRAINING_FILE)
    p.add_argument("--model", type=Path, default=MODEL_FILE)
    p.add_argument("--report", type=Path, default=REPORT_FILE)
    a = p.parse_args()
    print(json.dumps(train(a.data, a.model, a.report), indent=2))


if __name__ == "__main__":
    main()
