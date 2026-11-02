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

## `dataset_v2.csv` — SMOTE-oversampled training data

`dataset_v2.csv` is **not** a new raw source — it's derived entirely from
`dataset_v1.csv` via `healthy.data.resample.build_resampled_dataset`. It
contains 130,981 real rows (`dataset_v1.csv`'s 80% training split, same
`random_state=42` stratified split `healthy.model.train` already uses) plus
1,367 SMOTE-interpolated rows for the 30 classes that scored recall < 0.7 in
`models/diagnosis_model_v1.per_class_report.csv`. SMOTE only interpolates
between existing real rows of the same class — it never introduces new
disease/symptom correlations.

Every row carries an `is_synthetic` column (`False` = real, `True` =
SMOTE-generated) so real and synthetic rows stay distinguishable; it is
dropped before model fitting (`healthy.model.train`), not used as a feature.
**`dataset_v2.csv` contains no held-out test rows at all** — any model
trained on it (`models/diagnosis_model_v2.pkl`) must be evaluated against
`dataset_v1.csv`'s test split (`healthy.model.train`'s `test_dataset_path`
argument / `healthy.model.evaluate` pointed at `dataset_v1.csv`), never a
split of `dataset_v2.csv` itself.

Result (see `models/diagnosis_model_v{1,2}.per_class_report.csv`): recall on
the 30 targeted classes improved on average (0.59 -> 0.65), with roughly
half of them (e.g. uterine fibroids, ileus, diabetic retinopathy, glaucoma,
acne) improving substantially (+20-31pp) and the other half (mostly
psychiatric/dermatologic classes like psychotic disorder, depression,
corneal disorder, skin polyp) barely moving or regressing slightly. Overall
accuracy was unchanged (0.850 -> 0.851) and non-targeted classes' recall was
unaffected (0.892 -> 0.890). The classes SMOTE didn't help likely share
near-identical symptom-flag encodings with other diseases in the current
377-term vocabulary — a feature-overlap ceiling that more interpolated rows
of the same features can't fix; closing that gap would need genuinely new
symptom-level information (see the deferred DDXPlus mapping discussion).
