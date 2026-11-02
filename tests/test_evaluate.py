"""Unit tests for per-class model evaluation."""

import numpy as np
import pandas as pd

from healthy.model.evaluate import per_class_report, weak_classes


class _StubModel:
    """Hardcoded predictions: 'a' predicts perfectly, 'b' always misses."""

    classes_ = np.array(["a", "b"])

    def predict(self, X):
        # Always guesses 'a', so true 'b' rows are always misclassified.
        return np.array(["a", "a", "a", "a"])

    def predict_proba(self, X):
        # rows: 2 true 'a' (correctly top-ranked 'a'), 2 true 'b' (model ranks 'a' first)
        return np.array(
            [
                [0.9, 0.1],
                [0.9, 0.1],
                [0.9, 0.1],
                [0.9, 0.1],
            ]
        )


def test_per_class_report_computes_support_and_recall():
    X_test = pd.DataFrame({"f": [0, 0, 0, 0]})
    y_test = pd.Series(["a", "a", "b", "b"])

    report = per_class_report(_StubModel(), X_test, y_test, k=1)

    a_row = report.loc[report["disease"] == "a"].iloc[0]
    b_row = report.loc[report["disease"] == "b"].iloc[0]

    assert a_row["support"] == 2
    assert a_row["recall"] == 1.0
    assert b_row["support"] == 2
    assert b_row["recall"] == 0.0
    assert b_row["top_1_hit_rate"] == 0.0
    assert a_row["top_1_hit_rate"] == 1.0


def test_weak_classes_filters_by_recall_and_support():
    report = pd.DataFrame(
        {
            "disease": ["a", "b", "c"],
            "support": [10, 10, 1],
            "recall": [0.9, 0.2, 0.1],
        }
    )

    assert weak_classes(report, min_recall=0.5) == ["b", "c"]
    assert weak_classes(report, min_recall=0.5, min_support=5) == ["b"]
