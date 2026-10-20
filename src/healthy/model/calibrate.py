"""Post-hoc calibration of the diagnosis probability model (Requirement 3).

Takes the baseline classifier trained by ``healthy.model.train`` and wraps it
with a post-hoc calibration technique (Platt scaling / sigmoid, or isotonic
regression) so that a reported probability actually matches the empirical
frequency of being correct (an 87% score should be right ~87% of the time).

This module expects its own held-out calibration split — separate from both
the training data and the final test set used to report accuracy in
``train.py``. Reusing the test set for calibration would let the reported
"improvement" hide the fact that calibration was tuned against it.

Usage:
    python -m healthy.model.calibrate data/processed/dataset_v1.csv models/diagnosis_model_v1.pkl models/diagnosis_model_v1.calibrated.pkl
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd


def split_calibration_set(
    dataset_path: str | Path,
    test_size: float = 0.2,
    calib_size: float = 0.2,
    random_state: int = 42,
):
    """Split the processed dataset into train / calibration / test subsets.

    TODO: implement using ``sklearn.model_selection.train_test_split`` twice
    (stratified on the disease label both times, same as ``train.py``):
      1. Split off ``test_size`` as the final held-out test set — this
         should reproduce the exact same test rows ``train.train()`` already
         evaluated against (same ``test_size``/``random_state``), so the
         accuracy numbers stay comparable before/after calibration.
      2. From the remaining rows, split off ``calib_size`` (as a fraction of
         the *original* dataset, not of the remainder) as the calibration
         set; what's left is the training set.

    Returns:
        (X_train, X_calib, X_test, y_train, y_calib, y_test)
    """
    raise NotImplementedError


def calibrate(model, X_calib: pd.DataFrame, y_calib: pd.Series, method: str = "isotonic"):
    """Fit a post-hoc calibrator on top of an already-trained ``model``.

    TODO: implement using ``sklearn.calibration.CalibratedClassifierCV``.
    ``model`` is already fit, so it must be wrapped so ``CalibratedClassifierCV``
    treats it as frozen rather than re-fitting it per cross-validation fold:

        from sklearn.frozen import FrozenEstimator
        from sklearn.calibration import CalibratedClassifierCV

        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
        calibrated.fit(X_calib, y_calib)

    ``method`` is ``"isotonic"`` (nonparametric, needs more calibration data
    per class, more flexible) or ``"sigmoid"`` (Platt scaling, parametric,
    safer with fewer calibration examples per class).
    """
    raise NotImplementedError


def top1_confidence_and_correctness(model, X: pd.DataFrame, y: pd.Series):
    """Reduce a multiclass model's predictions to a binary-calibration view.

    A reliability diagram is inherently about binary confidence ("was the
    top prediction right, and how confident was the model?"), even though
    diagnosis prediction is multiclass. This collapses each case down to:
      - confidence: the model's top-1 predicted probability
      - correct: whether that top-1 prediction matched the true label

    TODO: implement using ``model.predict_proba(X)`` (take the max per row
    for confidence, and ``model.classes_[argmax]`` compared against ``y``
    for correctness).

    Returns:
        (correct: np.ndarray[bool], confidence: np.ndarray[float])
    """
    raise NotImplementedError


def compute_reliability_curve(correct, confidence, n_bins: int = 10):
    """Bin predictions by confidence and compute empirical accuracy per bin.

    TODO: implement using ``sklearn.calibration.calibration_curve(correct,
    confidence, n_bins=n_bins)``, which returns
    ``(fraction_of_positives, mean_predicted_value)`` per bin — i.e. the
    (x, y) points of a reliability diagram.
    """
    raise NotImplementedError


def compute_calibration_metrics(correct, confidence) -> dict:
    """Summarize miscalibration as scalars, to show "measurable improvement".

    TODO: implement at least:
      - Brier score: ``sklearn.metrics.brier_score_loss(correct, confidence)``
      - Expected Calibration Error (ECE): weighted average, across the same
        bins as ``compute_reliability_curve``, of
        ``|bin_accuracy - bin_mean_confidence|`` (no direct sklearn helper —
        this one you compute by hand from the binned values).

    Returns:
        dict, e.g. {"brier_score": ..., "ece": ...}
    """
    raise NotImplementedError


def plot_reliability_diagram(
    raw_curve,
    calibrated_curve,
    out_path: str | Path,
):
    """Plot raw vs. calibrated reliability curves against the ideal diagonal.

    TODO: implement with matplotlib (add it to requirements.txt):
      - x-axis: mean predicted confidence per bin
      - y-axis: empirical accuracy per bin
      - plot both ``raw_curve`` and ``calibrated_curve`` as line+markers
      - plot the ``y = x`` diagonal as the "perfectly calibrated" reference
      - save the figure to ``out_path`` (e.g. ``models/reliability_v1.png``)
    """
    raise NotImplementedError


def run_calibration(
    dataset_path: str | Path,
    model_path: str | Path,
    calibrated_model_out_path: str | Path,
    method: str = "isotonic",
):
    """End-to-end: split, calibrate, evaluate, and save the artifact + plot.

    TODO: wire the pieces above together:
      1. ``split_calibration_set`` on ``dataset_path``.
      2. Load the baseline model from ``model_path`` (``joblib.load``).
      3. Compute the raw reliability curve + metrics on ``X_test``/``y_test``
         using the *uncalibrated* model.
      4. ``calibrate`` the model on ``X_calib``/``y_calib``.
      5. Compute the calibrated reliability curve + metrics on the same
         ``X_test``/``y_test``.
      6. Print/compare both metrics dicts (this is the "measurable
         improvement" acceptance criterion).
      7. Save the calibrated model to ``calibrated_model_out_path`` and the
         reliability diagram alongside it (mirror ``train.py``'s pattern of
         saving a sibling file, e.g. ``.reliability.png``).
    """
    raise NotImplementedError


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python -m healthy.model.calibrate "
            "<dataset_path> <model_path> <calibrated_model_out_path>"
        )
        sys.exit(1)

    run_calibration(sys.argv[1], sys.argv[2], sys.argv[3])
