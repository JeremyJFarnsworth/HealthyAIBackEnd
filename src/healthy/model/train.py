"""Diagnosis probability model training.

Loads a versioned processed dataset from ``data/processed/`` (produced by
``healthy.data.clean.clean_diagnosis_dataset`` from
``final_symptoms_to_disease.csv``) and trains the diagnosis probability
classifier described in the proposal (Requirement 2), saving the resulting
model artifact to ``models/``. Calibration (Platt scaling / isotonic
regression, Requirement 3) is handled separately once this baseline
classifier exists.

Usage:
    python -m healthy.model.train data/processed/dataset_v1.csv models/diagnosis_model_v1.pkl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

# ``dataset_v1.csv`` already contains exactly the symptom flag columns plus
# the target, so leaving FEATURE_COLUMNS empty (use "every other column") is
# sufficient — no need to duplicate the 377-entry symptom vocabulary here.
FEATURE_COLUMNS: list[str] = []
TARGET_COLUMN: str = "disease"


def load_dataset(dataset_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the processed dataset and split into features/target."""
    df = pd.read_csv(dataset_path)
    X = df[FEATURE_COLUMNS] if FEATURE_COLUMNS else df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def train(dataset_path: str | Path, model_out_path: str | Path):
    """Train the diagnosis probability classifier and save the artifact.

    ``LogisticRegression`` with its default multinomial/softmax behavior for
    multi-class problems already satisfies Requirement 2's "scores sum to
    100%" acceptance criterion via ``predict_proba``, so it's used here as a
    fast baseline to validate the ingestion -> training -> calibration
    pipeline end-to-end. Swap in a neural classifier later (per the
    proposal's "deep learning classifier" language) if baseline accuracy
    turns out to be insufficient — the surrounding pipeline doesn't change.
    """
    from sklearn.linear_model import LogisticRegression

    X, y = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    top5_accuracy = top_k_accuracy_score(y_test, proba, k=5, labels=model.classes_)
    print(f"held-out accuracy: {accuracy:.4f}")
    print(f"held-out top-5 accuracy: {top5_accuracy:.4f}")

    model_out_path = Path(model_out_path)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out_path)

    # Saved alongside the model so inference code can map predict_proba's
    # column order back to disease names without re-deriving the class list.
    classes_path = model_out_path.with_suffix(".classes.json")
    classes_path.write_text(json.dumps(list(model.classes_)))

    return model


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m healthy.model.train <dataset_path> <model_out_path>")
        sys.exit(1)

    train(sys.argv[1], sys.argv[2])
