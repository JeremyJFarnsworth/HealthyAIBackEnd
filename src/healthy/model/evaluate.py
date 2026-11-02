"""Per-class performance profiling for the diagnosis probability classifier.

Global metrics (``train.py``'s held-out accuracy / top-5 accuracy) hide which
of the 254 disease classes the baseline actually struggles with. This module
reproduces the exact held-out split ``train.py`` uses, scores a saved model
against it, and reports precision/recall/f1/support/top-5 hit rate per class
so later effort (e.g. oversampling) can target the classes that need it
instead of guessing.

Usage:
    python -m healthy.model.evaluate data/processed/dataset_v1.csv models/diagnosis_model_v1.pkl [report_out_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from healthy.model.train import load_dataset


def _top_k_hit(y_test: pd.Series, proba: np.ndarray, classes: np.ndarray, k: int = 5) -> np.ndarray:
    """Per-row bool: whether the true label is among the top-k predicted classes."""
    top_k_idx = np.argsort(proba, axis=1)[:, -k:]
    top_k_labels = classes[top_k_idx]
    y_arr = y_test.to_numpy().reshape(-1, 1)
    return (top_k_labels == y_arr).any(axis=1)


def per_class_report(model, X_test: pd.DataFrame, y_test: pd.Series, k: int = 5) -> pd.DataFrame:
    """One row per class: support, precision, recall, f1, top-k hit rate."""
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = model.classes_

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, zero_division=0
    )

    hits = _top_k_hit(y_test, proba, classes, k=k)
    hit_by_class = pd.Series(hits, index=y_test.index).groupby(y_test).mean()

    report = pd.DataFrame(
        {
            "disease": classes,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            f"top_{k}_hit_rate": hit_by_class.reindex(classes).fillna(0.0).to_numpy(),
        }
    )
    return report


def weak_classes(report: pd.DataFrame, min_recall: float = 0.5, min_support: int = 0) -> list[str]:
    """Disease names falling below ``min_recall`` (with at least ``min_support`` test rows)."""
    mask = (report["recall"] < min_recall) & (report["support"] >= min_support)
    return report.loc[mask, "disease"].tolist()


def run_evaluation(
    dataset_path: str | Path,
    model_path: str | Path,
    report_out_path: str | Path | None = None,
    k: int = 5,
) -> pd.DataFrame:
    """Reproduce train.py's held-out split, score the model, print/save a per-class report."""
    X, y = load_dataset(dataset_path)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(model_path)
    report = per_class_report(model, X_test, y_test, k=k)
    report = report.sort_values("recall")

    print(report.to_string(index=False))

    if report_out_path is not None:
        report_out_path = Path(report_out_path)
        report_out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_out_path, index=False)

    return report


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python -m healthy.model.evaluate <dataset_path> <model_path> [report_out_path]")
        sys.exit(1)

    out_path = sys.argv[3] if len(sys.argv) == 4 else None
    run_evaluation(sys.argv[1], sys.argv[2], out_path)
