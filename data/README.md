# Data

This folder is git-ignored (except for the folder markers) — never commit raw
or processed patient data, even if it appears anonymized.

## Layout

- `raw/` — the dataset exactly as downloaded, untouched. Drop the downloaded
  dataset here before running ingestion.
- `processed/` — cleaned, de-identified, structured output produced by
  `src/healthy/data/ingest.py` and `src/healthy/data/clean.py`.

## Versioning convention

Each processed dataset is written as:

```
data/processed/dataset_v{N}.csv
```

where `{N}` increments for every distinct cleaning run that changes the
output (new source data, new cleaning logic, etc.). The version used to
train a given model is recorded alongside the model artifact in `models/`
(e.g. `models/diagnosis_model_v{N}.pkl` trained from `dataset_v{N}.csv`),
so any model can be traced back to the exact dataset version that produced
it.

## Raw source -> processed artifact mapping

`data/raw/` holds four unrelated datasets (different schemas, different
disease taxonomies). Each one feeds a different part of the system — they
are not merged:

| raw file | processed artifact | used for |
|---|---|---|
| `final_symptoms_to_disease.csv` | `dataset_v1.csv` (via `healthy.data.clean.clean_diagnosis_dataset`) | training the structured diagnosis probability classifier (Req 2/3) — chosen over `data.csv` because its 254 disease classes each have >=402 rows, vs. `data.csv`'s 713 classes with as few as 6 rows each |
| `data.csv` | `symptom_vocab_v1.json` (via `healthy.data.vocab`) | source of the canonical 377-symptom vocabulary only; `final_symptoms_to_disease.csv` is a filtered, text-encoded subset of this same data, so its rows aren't used directly |
| `Symptom2Disease.csv` | `text_cases_v1.csv` (via `healthy.data.clean.clean_text_cases`) | dev/eval set for the free-text -> structured-symptom extraction step of the LLM reasoning layer (Req 4); not used to train the classifier |
| `Healthcare.csv` | (none yet) | parked as a demo/smoke-test fixture for later UI/API work; symptom-disease relations look close to random, not used for training |
