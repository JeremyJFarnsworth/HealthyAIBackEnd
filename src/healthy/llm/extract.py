"""Free-text case description -> canonical symptom list (Requirement 4).

Converts a physician's free-text narrative into symptom names drawn from the
same 377-entry vocabulary (``healthy.data.vocab``) the diagnosis classifier
was trained on, so the result can be fed straight into
``healthy.data.clean.vectorize_symptom_text``.
"""

from __future__ import annotations

from healthy.llm.client import chat_completion_json
from healthy.llm.schemas import ExtractedSymptoms

_SYSTEM_PROMPT_TEMPLATE = """You are a clinical symptom extraction assistant.

Given a patient case description, identify which of the following symptoms \
are explicitly supported by the text. Only use symptoms from this exact list \
(use the exact wording given), and only include a symptom if the text clearly \
describes it -- do not guess or infer symptoms that aren't mentioned.

Symptom list:
{vocab_list}

Respond with a JSON object: {{"symptoms": [...]}} containing only symptoms \
from the list above that apply to this case."""


def extract_symptoms(case_text: str, vocab: list[str]) -> list[str]:
    """Return the subset of ``vocab`` (underscore form) the LLM found in ``case_text``.

    ``vocab`` entries use underscores; the LLM is prompted with the
    space-separated display form (matching how symptoms read in natural
    text), and any term it returns that isn't an exact match for a vocab
    entry is silently dropped -- the LLM is not trusted to stay within the
    list, the same way ``MultiLabelBinarizer`` silently drops unknown labels
    elsewhere in this pipeline.
    """
    display_vocab = [name.replace("_", " ") for name in vocab]
    display_to_vocab = dict(zip(display_vocab, vocab))

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(vocab_list=", ".join(display_vocab))
    response = chat_completion_json(system_prompt, case_text, ExtractedSymptoms)
    extracted = ExtractedSymptoms.model_validate(response)

    return [display_to_vocab[s] for s in extracted.symptoms if s in display_to_vocab]
