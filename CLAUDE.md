# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Backend for Healthy, an AI diagnostic assistant: submit a patient case (symptoms,
text, images, files) and get back ranked candidate diagnoses with calibrated
probability scores and treatment plans. This repo covers the data pipeline and
diagnosis probability model now, with the API, calibration, and LLM reasoning
layer planned for later weeks (see `Resouces/ProjectProposal.docx`). Current
status: Week 1-2 focus — dataset cleaning and an initial baseline classifier.

## Commands

```bash
# setup
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# run all tests
pytest

# run a single test
pytest tests/test_clean.py::test_vectorize_symptom_text_marks_present_symptoms

# pipeline, in order (raw -> vocab -> processed dataset -> model)
python -m healthy.data.vocab data/raw/data.csv data/processed/symptom_vocab_v1.json
python -m healthy.data.clean diagnosis data/raw/final_symptoms_to_disease.csv data/processed/symptom_vocab_v1.json data/processed/dataset_v1.csv
python -m healthy.data.clean text-cases data/raw/Symptom2Disease.csv data/processed/text_cases_v1.csv
python -m healthy.model.train data/processed/dataset_v1.csv models/diagnosis_model_v1.pkl
```

`conftest.py` puts `src/` on `sys.path` so `healthy` imports directly in tests
without an editable install.

## Architecture

### Two-pipeline data strategy

`data/raw/` holds four unrelated datasets with different schemas and disease
taxonomies — they are deliberately never merged. Each raw file maps to exactly
one processed artifact and one downstream use (full mapping and rationale in
`data/README.md`):

- `final_symptoms_to_disease.csv` -> `dataset_v1.csv` (via
  `healthy.data.clean.clean_diagnosis_dataset`) — trains the structured
  diagnosis probability classifier. Chosen over `data.csv` because its 254
  disease classes each have >=402 rows, vs. `data.csv`'s 713 classes with as
  few as 6 rows.
- `data.csv` -> `symptom_vocab_v1.json` (via `healthy.data.vocab`) — source of
  the canonical 377-symptom vocabulary only; `final_symptoms_to_disease.csv`'s
  rows are a filtered, text-encoded subset of this same data and aren't used
  directly.
- `Symptom2Disease.csv` -> `text_cases_v1.csv` (via
  `healthy.data.clean.clean_text_cases`) — free-text dev/eval set for the
  future LLM reasoning layer's free-text -> structured-symptom extraction;
  never used to train the classifier.
- `Healthcare.csv` -> parked as a demo/smoke-test fixture for later UI/API
  work (symptom-disease relations look close to random).

### Vocabulary as the shared contract

`symptom_vocab_v1.json` (377 symptom names, from `data.csv`'s header minus the
label column) is the single source of truth for feature order. Both the
training dataset's binary flag columns and the trained model's
`predict_proba` column order are built against this one vocabulary — anything
that later produces a symptom vector (free-text extraction, a new raw
dataset, API input validation) must stay consistent with it via
`healthy.data.vocab.load_vocab`.

### Dataset versioning

Processed datasets are written as `data/processed/dataset_v{N}.csv`, where
`{N}` increments for any cleaning-run change (new source data, new cleaning
logic). A trained model is named to match the dataset version that produced
it (e.g. `models/diagnosis_model_v{N}.pkl` <- `dataset_v{N}.csv`), so a model
can always be traced back to its exact training data. `data/` and `models/`
are both git-ignored — raw or processed patient data must never be committed,
even if it appears anonymized.

### Model baseline

`healthy.model.train` uses `LogisticRegression` (multinomial/softmax) as a
fast baseline: `predict_proba` already satisfies the "scores sum to 100%"
requirement, letting the ingestion -> training -> calibration pipeline be
validated end-to-end before swapping in a heavier classifier. Calibration
(Platt scaling / isotonic regression) is a separate later step, not yet
implemented, and is expected to wrap this same baseline rather than replace
the surrounding pipeline.

### De-identification

`healthy.data.clean.IDENTIFIER_COLUMNS` is the declared list of direct
patient-identifier columns to drop. It's empty today because neither
training source (`final_symptoms_to_disease.csv`, `Symptom2Disease.csv`)
carries a direct identifier — they're symptom/diagnosis knowledge-base rows,
not individual patient records. Any future raw source that does carry
identifiers (e.g. `Healthcare.csv`'s `Patient_ID`) should be declared here
before it's used for training.
