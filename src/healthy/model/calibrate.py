"""Post-hoc calibration of the diagnosis probability model (Requirement 3).

Takes the baseline classifier trained by ``healthy.model.train`` and wraps it
with temperature scaling (a single global scalar on the logits before
softmax) so that a reported probability actually matches the empirical
frequency of being correct (an 87% score should be right ~87% of the time).

NOTE ON DEVIATION FROM THE PROPOSAL: the proposal specifies Platt scaling
(sigmoid) or isotonic regression. Both were implemented and evaluated against
temperature scaling on dataset_v1.csv / diagnosis_model_v1.pkl, and both
measurably *hurt* calibration (ECE roughly 8-10x worse than raw) because
CalibratedClassifierCV fits 254 independent one-vs-rest curves and
renormalizes across classes, which distorts top-1 confidence. Temperature
scaling was the only method that didn't regress the model, so it's the only
one implemented here; see commit history for the isotonic/sigmoid comparison.

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
    from sklearn.model_selection import train_test_split

    from healthy.model.train import load_dataset

    X, y = load_dataset(dataset_path)

    X_train_calib, X_test, y_train_calib, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # calib_size is a fraction of the *original* dataset; re-express it as a
    # fraction of the remainder (what's left after removing the test split).
    remaining_calib_size = calib_size / (1 - test_size)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_calib,
        y_train_calib,
        test_size=remaining_calib_size,
        random_state=random_state,
        stratify=y_train_calib,
    )

    return X_train, X_calib, X_test, y_train, y_calib, y_test


class _TemperatureScaledModel:
    """Wraps a fitted model, rescaling its logits by a learned scalar temperature before softmax.

    Unlike ``CalibratedClassifierCV``'s one-vs-rest approach (254 independent
    per-class curves, remixed via renormalization), this reshapes the *whole*
    probability vector with a single number: T > 1 softens overconfident
    predictions toward uniform, T < 1 sharpens underconfident ones, T == 1 is
    a no-op. Because it's one global monotonic rescaling, it can never change
    which class ends up as the top-1 prediction — only how confident the
    model is allowed to say it is about that choice.
    """

    def __init__(self, model, temperature: float):
        self.model = model
        self.temperature = temperature
        self.classes_ = model.classes_

    def predict_proba(self, X):
        import numpy as np

        logits = self.model.decision_function(X)
        if logits.ndim == 1:
            logits = np.column_stack([-logits, logits])

        scaled = logits / self.temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_scaled = np.exp(scaled)
        return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def _fit_temperature(model, X_calib: pd.DataFrame, y_calib: pd.Series) -> float:
    """Find the scalar T that minimizes calibration-set log-loss after rescaling."""
    import numpy as np
    from scipy.optimize import minimize_scalar

    logits = model.decision_function(X_calib)
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])

    class_to_idx = {label: idx for idx, label in enumerate(model.classes_)}
    y_idx = np.array([class_to_idx[label] for label in y_calib])

    def negative_log_likelihood(log_temperature: float) -> float:
        temperature = np.exp(log_temperature)
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(len(y_idx)), y_idx].mean()

    # Optimize in log-space so the search is symmetric around T=1 (no-op) and
    # T is guaranteed positive; bounds correspond to T in [~0.05, ~20].
    result = minimize_scalar(negative_log_likelihood, bounds=(-3, 3), method="bounded")
    return float(np.exp(result.x))


def calibrate(model, X_calib: pd.DataFrame, y_calib: pd.Series):
    """Fit temperature scaling on top of an already-trained ``model``.

    Rescales the model's logits by a single learned scalar temperature before
    softmax (see ``_TemperatureScaledModel``/``_fit_temperature``). Unlike
    per-class approaches (isotonic/Platt via ``CalibratedClassifierCV``),
    this can't distort which class wins, only how confident the model is
    about it — isotonic and sigmoid were tried and measurably hurt
    calibration on this model (see module docstring), so they aren't offered
    here.
    """
    temperature = _fit_temperature(model, X_calib, y_calib)
    return _TemperatureScaledModel(model, temperature)


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
    import numpy as np

    proba = model.predict_proba(X)
    top_idx = proba.argmax(axis=1)
    confidence = proba[np.arange(len(proba)), top_idx]
    predicted = np.asarray(model.classes_)[top_idx]
    correct = predicted == np.asarray(y)
    return correct, confidence


def compute_reliability_curve(correct, confidence, n_bins: int = 10):
    """Bin predictions by confidence and compute empirical accuracy per bin.

    TODO: implement using ``sklearn.calibration.calibration_curve(correct,
    confidence, n_bins=n_bins)``, which returns
    ``(fraction_of_positives, mean_predicted_value)`` per bin — i.e. the
    (x, y) points of a reliability diagram.
    """
    from sklearn.calibration import calibration_curve

    return calibration_curve(correct, confidence, n_bins=n_bins)


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
    import numpy as np
    from sklearn.metrics import brier_score_loss

    brier = brier_score_loss(correct, confidence)

    # ECE: same equal-width confidence bins as calibration_curve, but we need
    # each bin's *sample count* too (to weight it), which calibration_curve
    # doesn't return, so the binning is redone by hand here.
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidence, bin_edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    n_total = len(confidence)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_accuracy = correct[mask].mean()
        bin_confidence = confidence[mask].mean()
        bin_weight = mask.sum() / n_total
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return {"brier_score": brier, "ece": ece}


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
    import matplotlib.pyplot as plt

    raw_accuracy, raw_confidence = raw_curve
    calibrated_accuracy, calibrated_confidence = calibrated_curve

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(raw_confidence, raw_accuracy, marker="o", label="Raw (uncalibrated)")
    ax.plot(calibrated_confidence, calibrated_accuracy, marker="o", label="Calibrated")

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run_calibration(
    dataset_path: str | Path,
    model_path: str | Path,
    calibrated_model_out_path: str | Path,
):
    """End-to-end: split, calibrate, evaluate, and save the artifact + plot.

    Splits off the calibration/test sets, computes raw reliability
    curve/metrics on the uncalibrated model, fits temperature scaling on the
    calibration set, recomputes the same curve/metrics on the calibrated
    model, prints both for comparison, then saves the calibrated model and
    its reliability diagram (as a sibling ``.reliability.png`` file).
    """
    _X_train, X_calib, X_test, _y_train, y_calib, y_test = split_calibration_set(dataset_path)

    model = joblib.load(model_path)

    raw_correct, raw_confidence = top1_confidence_and_correctness(model, X_test, y_test)
    raw_curve = compute_reliability_curve(raw_correct, raw_confidence)
    raw_metrics = compute_calibration_metrics(raw_correct, raw_confidence)

    calibrated_model = calibrate(model, X_calib, y_calib)

    calibrated_correct, calibrated_confidence = top1_confidence_and_correctness(
        calibrated_model, X_test, y_test
    )
    calibrated_curve = compute_reliability_curve(calibrated_correct, calibrated_confidence)
    calibrated_metrics = compute_calibration_metrics(calibrated_correct, calibrated_confidence)

    print(f"raw:        {raw_metrics}")
    print(f"calibrated: {calibrated_metrics}")

    calibrated_model_out_path = Path(calibrated_model_out_path)
    calibrated_model_out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, calibrated_model_out_path)

    reliability_out_path = calibrated_model_out_path.with_suffix(".reliability.png")
    plot_reliability_diagram(raw_curve, calibrated_curve, reliability_out_path)

    return calibrated_model


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python -m healthy.model.calibrate "
            "<dataset_path> <model_path> <calibrated_model_out_path>"
        )
        sys.exit(1)

    run_calibration(sys.argv[1], sys.argv[2], sys.argv[3])
