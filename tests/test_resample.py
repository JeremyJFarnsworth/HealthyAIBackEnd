"""Unit tests for SMOTE oversampling of weak classes."""

import pandas as pd

from healthy.data.resample import build_resampled_dataset, oversample_weak_classes


def _toy_data():
    # 10 'a', 10 'b', 2 'c' — 'c' is the minority class.
    X = pd.DataFrame(
        {
            "f1": list(range(10)) + list(range(10)) + [0, 1],
            "f2": list(range(10)) + list(range(10)) + [1, 0],
        }
    )
    y = pd.Series(["a"] * 10 + ["b"] * 10 + ["c"] * 2)
    return X, y


def test_oversample_weak_classes_only_changes_targeted_class():
    X, y = _toy_data()

    X_resampled, y_resampled, is_synthetic = oversample_weak_classes(
        X, y, target_classes=["c"], k_neighbors=1
    )

    counts = y_resampled.value_counts()
    assert counts["a"] == 10
    assert counts["b"] == 10
    assert counts["c"] > 2


def test_oversample_weak_classes_flags_synthetic_rows_correctly():
    X, y = _toy_data()

    X_resampled, y_resampled, is_synthetic = oversample_weak_classes(
        X, y, target_classes=["c"], k_neighbors=1
    )

    n_original = len(X)
    assert not is_synthetic.iloc[:n_original].any()
    assert is_synthetic.iloc[n_original:].all()
    assert len(is_synthetic) == len(y_resampled) == len(X_resampled)
    # All synthetic rows belong to the targeted class.
    synthetic_labels = y_resampled[is_synthetic]
    assert set(synthetic_labels) == {"c"}


def test_build_resampled_dataset_round_trips_and_preserves_columns(tmp_path):
    # Needs enough rows per class that an 80/20 split still leaves SMOTE
    # room to find neighbors in the training portion.
    X = pd.DataFrame(
        {
            "f1": list(range(20)) + list(range(20)) + list(range(10)),
            "f2": list(range(20)) + list(range(20)) + list(range(10)),
        }
    )
    y = pd.Series(["a"] * 20 + ["b"] * 20 + ["c"] * 10)
    in_df = pd.concat([y.rename("disease"), X], axis=1)
    in_path = tmp_path / "dataset_v1.csv"
    in_df.to_csv(in_path, index=False)

    out_path = tmp_path / "dataset_v2.csv"
    result = build_resampled_dataset(
        in_path, out_path, target_classes=["c"], k_neighbors=3
    )

    assert out_path.exists()
    on_disk = pd.read_csv(out_path)
    assert list(on_disk.columns) == ["disease", "f1", "f2", "is_synthetic"]
    assert list(result.columns) == ["disease", "f1", "f2", "is_synthetic"]
    # Output is training-only: never as many rows as the full input dataset
    # would have if the test split had been included.
    assert len(result) < len(in_df) + sum(result["is_synthetic"])
