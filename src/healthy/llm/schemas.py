"""Structured-output contracts for the LLM reasoning layer (Requirement 4).

Both LLM calls in ``healthy.llm.reasoning`` must return JSON matching one of
these models; a response that doesn't validate raises ``pydantic.ValidationError``
rather than being passed through as free-form text, per Requirement 4's "the
system cannot return an unstructured or free-form response" constraint.
"""

from __future__ import annotations

from pydantic import BaseModel


class ExtractedSymptoms(BaseModel):
    """Output of the free-text -> canonical-symptom extraction LLM call."""

    symptoms: list[str]


class LLMDiagnosisDetail(BaseModel):
    """LLM-authored detail for one diagnosis, as returned by the second LLM call.

    Deliberately excludes ``score`` -- the probability comes from the
    calibrated classifier, not the LLM, and is merged in afterward by
    ``healthy.llm.reasoning`` so the final score is never something the LLM
    could invent or alter.
    """

    diagnosis: str
    description: str
    common_symptoms: list[str]
    treatment_plan: str
    suggested_tests: list[str]


class LLMDiagnosisDetails(BaseModel):
    """Wrapper for the second LLM call's response: detail for each candidate diagnosis."""

    details: list[LLMDiagnosisDetail]


class DiagnosisDetail(BaseModel):
    """One ranked diagnosis, combining the classifier's score with LLM-authored detail."""

    diagnosis: str
    score: float
    description: str
    common_symptoms: list[str]
    treatment_plan: str
    suggested_tests: list[str]


class DiagnosisResponse(BaseModel):
    """Final structured response returned by ``diagnose_from_text``."""

    results: list[DiagnosisDetail]
