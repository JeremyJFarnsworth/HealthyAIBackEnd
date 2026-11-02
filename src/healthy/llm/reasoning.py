"""End-to-end LLM reasoning layer: free text -> structured diagnosis output (Requirement 4).

Pipeline: free-text case -> extract canonical symptoms -> vectorize against
the training vocabulary -> calibrated classifier's probability distribution
-> LLM fills in description/treatment/tests for the top candidates -> merge
into one structured response. The classifier, not the LLM, is the sole
source of the probability scores.

Usage:
    python -m healthy.llm.reasoning "<free-text case description>"
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

from healthy.data.clean import vectorize_symptom_text
from healthy.data.vocab import load_vocab
from healthy.llm.client import chat_completion_json
from healthy.llm.extract import extract_symptoms
from healthy.llm.schemas import DiagnosisDetail, DiagnosisResponse, LLMDiagnosisDetails

DEFAULT_MODEL_PATH = "models/diagnosis_model_v1.calibrated.pkl"
DEFAULT_VOCAB_PATH = "data/processed/symptom_vocab_v1.json"

_DETAIL_SYSTEM_PROMPT = """You are a clinical decision-support assistant.

A patient case has been run through a calibrated diagnosis classifier, which \
produced the following ranked candidate diagnoses with probability scores:

{candidates}

For EACH candidate diagnosis listed above, provide: a brief description of \
the disease, its 3 most common symptoms, a recommended treatment plan, and a \
list of suggested confirmatory tests. Base this on the patient case text \
provided by the user and general medical knowledge.

Respond with a JSON object: {{"details": [...]}} with exactly one entry per \
candidate diagnosis listed above, in the same order, using the exact \
diagnosis names given."""


def _top_n_candidates(
    model, X: pd.DataFrame, top_n: int
) -> list[tuple[str, float]]:
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, proba), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_n]


def diagnose_from_text(
    case_text: str,
    top_n: int = 5,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    vocab_path: str | Path = DEFAULT_VOCAB_PATH,
) -> dict:
    """Run the full free-text -> structured-diagnosis pipeline for one case."""
    vocab = load_vocab(vocab_path)
    symptoms = extract_symptoms(case_text, vocab)

    symptom_text = pd.Series([", ".join(s.replace("_", " ") for s in symptoms)])
    X = vectorize_symptom_text(symptom_text, vocab)

    model = joblib.load(model_path)
    candidates = _top_n_candidates(model, X, top_n)

    candidates_text = "\n".join(
        f"- {diagnosis}: {score * 100:.1f}%" for diagnosis, score in candidates
    )
    response = chat_completion_json(
        _DETAIL_SYSTEM_PROMPT.format(candidates=candidates_text),
        case_text,
        LLMDiagnosisDetails,
    )
    llm_details = LLMDiagnosisDetails.model_validate(response)

    details_by_diagnosis = {d.diagnosis: d for d in llm_details.details}

    results = []
    for diagnosis, score in candidates:
        detail = details_by_diagnosis.get(diagnosis)
        if detail is None:
            continue
        results.append(
            DiagnosisDetail(
                diagnosis=diagnosis,
                score=float(score),
                description=detail.description,
                common_symptoms=detail.common_symptoms,
                treatment_plan=detail.treatment_plan,
                suggested_tests=detail.suggested_tests,
            )
        )

    return DiagnosisResponse(results=results).model_dump()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python -m healthy.llm.reasoning "<free-text case description>"')
        sys.exit(1)

    import json

    print(json.dumps(diagnose_from_text(sys.argv[1]), indent=2))
