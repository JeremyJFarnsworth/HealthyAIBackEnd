"""Unit tests for the diagnosis-dataset symptom vectorizer."""

import pandas as pd

from healthy.data.clean import _vectorize_symptom_text


def test_vectorize_symptom_text_marks_present_symptoms():
    vocab = ["skin_rash", "joint_pain", "fever"]
    symptom_text = pd.Series(["skin rash, fever", "joint pain"])

    features = _vectorize_symptom_text(symptom_text, vocab)

    assert list(features.columns) == vocab
    assert features.loc[0].tolist() == [1, 0, 1]
    assert features.loc[1].tolist() == [0, 1, 0]


def test_vectorize_symptom_text_ignores_unknown_symptoms():
    vocab = ["fever"]
    symptom_text = pd.Series(["fever, some unknown symptom"])

    features = _vectorize_symptom_text(symptom_text, vocab)

    assert features.loc[0].tolist() == [1]
