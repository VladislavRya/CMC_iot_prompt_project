from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import LABEL_COL, find_csv, load_dataset  # noqa: E402


def build_baseline_pipeline(X: pd.DataFrame) -> Pipeline:
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    clf = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.9,
        random_state=42,
    )

    return Pipeline([("prep", preprocessor), ("clf", clf)])


def build_rf_baseline(X: pd.DataFrame) -> Pipeline:
    pipe = build_baseline_pipeline(X)
    pipe.steps[-1] = (
        "clf",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    )
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand-coded IDS baseline")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "hand_coded_model.joblib"
    )
    parser.add_argument("--model", choices=["gb", "rf"], default="gb")
    args = parser.parse_args()

    csv_path = find_csv(args.data_dir)
    print(f"Loading: {csv_path}")
    X, y = load_dataset(args.data_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    use_rf = args.model == "rf" or len(X) > 100_000
    if use_rf and args.model != "rf":
        print(f"Large dataset ({len(X)} rows): using RandomForest instead of GradientBoosting")

    pipeline = build_rf_baseline(X_train) if use_rf else build_baseline_pipeline(X_train)

    cv = StratifiedKFold(n_splits=3 if len(X) > 100_000 else 5, shuffle=True, random_state=42)
    cv_data = X_train
    cv_labels = y_train
    if len(X_train) > 100_000:
        cv_idx = (
            pd.concat([y_train], axis=1)
            .groupby(LABEL_COL, group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), 25_000), random_state=42))
            .index
        )
        cv_data = X_train.loc[cv_idx]
        cv_labels = y_train.loc[cv_idx]
        print(f"CV on stratified subsample: {len(cv_data)} rows")
    cv_scores = cross_val_score(
        pipeline, cv_data, cv_labels, cv=cv, scoring="f1", n_jobs=1
    )
    print(f"5-fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "model": "rf" if use_rf else args.model,
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\n=== Hand-coded baseline (test) ===")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k}: {metrics[k]:.4f}")
    print("Confusion matrix:\n", np.array(metrics["confusion_matrix"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output)


if __name__ == "__main__":
    main()
