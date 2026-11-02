"""SMOTE oversampling for underperforming diagnosis classes.

Interpolates *only* the real rows already present for a targeted class — it
never invents new clinical facts, unlike free-form synthetic data generation.
Every output row is tagged via ``is_synthetic`` so real and synthetic rows
stay distinguishable downstream (see ``data/README.md`` for the resulting
``dataset_v2.csv``'s documented provenance).

Operates only on ``train.py``'s training split (SMOTE must never touch
held-out evaluation rows): this module reproduces that exact 80/20 stratified
split, discards the test portion, and oversamples within the training rows
only. Evaluation always goes back to ``dataset_v1.csv``'s test split via
``evaluate.py`` / ``train.py``'s ``test_dataset_path`` argument.

Usage:
    python -m healthy.data.resample data/processed/dataset_v1.csv data/processed/dataset_v2.csv [target_class ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from healthy.model.train import TARGET_COLUMN, load_dataset


def oversample_weak_classes(
    X: pd.DataFrame,
    y: pd.Series,
    target_classes: list[str] | None = None,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """SMOTE-oversample only ``target_classes`` (default: classes below the median count).

    Returns ``(X_resampled, y_resampled, is_synthetic)`` where ``is_synthetic``
    is aligned to the resampled rows: ``False`` for every original row,
    ``True`` for each SMOTE-interpolated row.
    """
    counts = y.value_counts()

    if target_classes is None:
        median_count = counts.median()
        target_classes = counts[counts < median_count].index.tolist()

    target_classes = [c for c in target_classes if c in counts.index]
    if not target_classes:
        is_synthetic = pd.Series(False, index=y.index)
        return X.reset_index(drop=True), y.reset_index(drop=True), is_synthetic.reset_index(drop=True)

    median_count = int(counts.median())
    sampling_strategy = {
        cls: max(counts[cls], median_count) for cls in target_classes
    }

    # SMOTE needs at least k_neighbors+1 samples in a class; shrink k for tiny classes.
    smallest_targeted = min(counts[cls] for cls in target_classes)
    effective_k = min(k_neighbors, smallest_targeted - 1)
    effective_k = max(effective_k, 1)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=effective_k,
        random_state=random_state,
    )
    X_resampled, y_resampled = smote.fit_resample(X, y)

    n_original = len(X)
    is_synthetic = pd.Series(
        [False] * n_original + [True] * (len(X_resampled) - n_original)
    )

    return X_resampled.reset_index(drop=True), y_resampled.reset_index(drop=True), is_synthetic


def build_resampled_dataset(
    in_dataset_path: str | Path,
    out_dataset_path: str | Path,
    target_classes: list[str] | None = None,
    **smote_kwargs,
) -> pd.DataFrame:
    """Build dataset_v2.csv: dataset_v1.csv's training rows only, SMOTE-oversampled.

    Reproduces train.py's exact 80/20 stratified split, keeps only the 80%
    training rows, and oversamples weak classes within them. The output never
    contains any row from the held-out test split.
    """
    X, y = load_dataset(in_dataset_path)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_resampled, y_resampled, is_synthetic = oversample_weak_classes(
        X_train, y_train, target_classes=target_classes, **smote_kwargs
    )

    out_df = pd.concat(
        [y_resampled.rename(TARGET_COLUMN), X_resampled, is_synthetic.rename("is_synthetic")],
        axis=1,
    )

    out_dataset_path = Path(out_dataset_path)
    out_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dataset_path, index=False)

    return out_df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python -m healthy.data.resample <in_dataset_path> <out_dataset_path> [target_class ...]"
        )
        sys.exit(1)

    targets = sys.argv[3:] or None
    build_resampled_dataset(sys.argv[1], sys.argv[2], target_classes=targets)
