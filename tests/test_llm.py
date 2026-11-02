"""Unit tests for the LLM reasoning layer. The LLM and classifier are mocked --
these tests never make a network call or require LM Studio to be running."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from healthy.llm.extract import extract_symptoms
from healthy.llm.reasoning import diagnose_from_text
from healthy.llm.schemas import DiagnosisDetail


def test_extract_symptoms_filters_hallucinated_terms():
    vocab = ["skin_rash", "joint_pain", "fever"]

    with patch("healthy.llm.extract.chat_completion_json") as mock_chat:
        mock_chat.return_value = {
            "symptoms": ["skin rash", "fever", "something the model made up"]
        }
        result = extract_symptoms("my skin is itchy and red, and I have a fever", vocab)

    assert result == ["skin_rash", "fever"]


class _StubModel:
    classes_ = np.array(["Psoriasis", "Chickenpox"])

    def predict_proba(self, X):
        return np.array([[0.7, 0.3]])


def test_diagnose_from_text_assembles_response_with_classifier_scores():
    with (
        patch("healthy.llm.reasoning.load_vocab", return_value=["skin_rash", "fever"]),
        patch("healthy.llm.reasoning.extract_symptoms", return_value=["skin_rash"]),
        patch("healthy.llm.reasoning.joblib.load", return_value=_StubModel()),
        patch("healthy.llm.reasoning.chat_completion_json") as mock_chat,
    ):
        mock_chat.return_value = {
            "details": [
                {
                    "diagnosis": "Psoriasis",
                    "description": "A chronic skin condition.",
                    "common_symptoms": ["skin rash", "itching"],
                    "treatment_plan": "Topical corticosteroids.",
                    "suggested_tests": ["Skin biopsy"],
                },
                {
                    "diagnosis": "Chickenpox",
                    "description": "A viral infection causing an itchy rash.",
                    "common_symptoms": ["skin rash", "fever"],
                    "treatment_plan": "Supportive care, antihistamines.",
                    "suggested_tests": ["Clinical exam"],
                },
            ]
        }

        response = diagnose_from_text("I have a skin rash", top_n=2)

    assert [r["diagnosis"] for r in response["results"]] == ["Psoriasis", "Chickenpox"]
    assert response["results"][0]["score"] == pytest.approx(0.7)
    assert response["results"][1]["score"] == pytest.approx(0.3)
    assert response["results"][0]["treatment_plan"] == "Topical corticosteroids."


def test_malformed_llm_response_raises_validation_error():
    with pytest.raises(ValidationError):
        DiagnosisDetail.model_validate(
            {
                "diagnosis": "Psoriasis",
                "score": 0.7,
                "description": "A chronic skin condition.",
                # missing common_symptoms, treatment_plan, suggested_tests
            }
        )
