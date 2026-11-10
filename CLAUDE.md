# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Backend for Healthy, an AI diagnostic assistant: submit a patient case (symptoms,
text, images, files) and get back ranked candidate diagnoses with calibrated
probability scores and treatment plans. This repo covers the data pipeline,
diagnosis probability model, calibration, LLM reasoning layer, and a FastAPI
persistence/auth layer (see `Resouces/ProjectProposal.docx`). Current status:
Week 5 complete — role-based auth and chat-history persistence on top of the
Week 1-4 model/LLM work. The Next.js frontend and website (weeks 6-8) are not
yet started.

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

# run the API (requires JWT_SECRET_KEY; DATABASE_URL defaults to a local sqlite file)
set JWT_SECRET_KEY=change-me   # cmd.exe; use $env:JWT_SECRET_KEY="change-me" in PowerShell
uvicorn healthy.api.app:app --reload --app-dir src
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

`dataset_v2.csv` is the first version carrying an `is_synthetic` provenance
column (`False`/`True`) — it is `dataset_v1.csv`'s training split with
SMOTE-oversampled rows added for underperforming classes identified via
`healthy.model.evaluate`'s per-class report (see `data/README.md` for full
rationale and results). It contains no held-out test rows at all, so any
model trained on it (`diagnosis_model_v2.pkl`, via `train()`'s
`test_dataset_path` argument) must be evaluated against `dataset_v1.csv`'s
real test split, never a split of `dataset_v2.csv` itself.

### Model baseline

`healthy.model.train` uses `LogisticRegression` (multinomial/softmax) as a
fast baseline: `predict_proba` already satisfies the "scores sum to 100%"
requirement, letting the ingestion -> training -> calibration pipeline be
validated end-to-end before swapping in a heavier classifier. Calibration
(temperature scaling, see `healthy.model.calibrate`) wraps this same baseline
rather than replacing the surrounding pipeline.

### Persistence, auth, and role-based access (Week 5)

Three new packages sit alongside `data/`/`model`/`llm`, each a separate
concern: `db/` (SQLAlchemy plumbing only — engine, session, ORM models),
`auth/` (credentials/tokens/roles, importable without FastAPI), and `api/`
(the only layer that imports FastAPI and the only caller of
`healthy.llm.reasoning.diagnose_from_text`). `diagnose_from_text` itself is
untouched by this work — it remains a pure, stateless function used
identically by the CLI and the `POST /chat` route; persistence happens only
in that route handler, not inside `healthy.llm`.

Config (`healthy.config.Settings`, via `pydantic-settings`) reads
`DATABASE_URL`, `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `JWT_EXPIRE_MINUTES` from
the environment — same convention as `llm/client.py`'s `LLM_BASE_URL`/
`LLM_MODEL`. `JWT_SECRET_KEY` has no default and must be set before importing
`healthy.config`, including in tests (see `tests/conftest.py`).

**Schema**: `User` (username, password_hash, role) has one `User.role` string
field (`"user"` or `"admin"`, validated as a `Literal` at the pydantic layer
rather than a native DB enum, to avoid enum-migration friction). `Conversation`
holds one submitted case (`case_text`, `user_id`) and has many `ChatMessage`
rows (one `"user"` message = the case text, one `"assistant"` message = the
JSON-dumped `DiagnosisResponse`). This is one `Conversation` per submitted
case rather than a flat per-user message log — it matches the natural
"list history -> open one conversation" UX and leaves room for future
multi-turn follow-ups without a schema change. Deleting a `User` cascades
(`cascade="all, delete-orphan"`) to delete their `Conversation`/`ChatMessage`
rows — this is how admin account deletion removes "all of their data" with no
separate manual cleanup step.

**Auth**: `passlib[bcrypt]` for hashing (pin `bcrypt==4.0.1` in
requirements.txt — passlib 1.7.4 is incompatible with bcrypt 4.1+/5.x's
removed `__about__` attribute and stricter 72-byte check used during its own
backend self-test; this is a known passlib/bcrypt version issue, not
something specific to this codebase). `python-jose[cryptography]` for JWTs,
HS256, no refresh tokens. Registration (`POST /auth/register`) is self-serve
and always creates `role="user"` — admin-only account creation has a
chicken-and-egg bootstrap problem, so **the first admin must be promoted
manually** (e.g. `UPDATE users SET role='admin' WHERE username=...` against
the configured database) after registering normally. This is a known
simplification for the capstone timeline, not production-ready as-is.

**Role-based access control**: `auth/deps.py`'s `get_current_user` (valid JWT
required) and `require_admin` (role must be `"admin"`) are FastAPI
dependencies, not checks scattered through handlers. `GET/GET-detail /chat`
filter by `Conversation.user_id == current_user.id` regardless of role — this
ownership filter *is* the server-side enforcement that a user can't view
another user's history (requirement 11); there is no "admin views any user's
history" endpoint yet. Admin's `DELETE /admin/users/{id}` rejects deleting a
target whose role is `"admin"`, and is a single-user-targeted delete only (no
bulk purge endpoint).

**No Alembic**: schema is created via `Base.metadata.create_all` at API
startup (`api/app.py`'s lifespan handler), not migrated. Appropriate for a
single developer with a still-evolving schema and no production data to
preserve yet — revisit once there's a deployed instance with real user data.

**Deferred / explicitly out of scope**: refresh tokens, rate limiting, email
verification, an admin "view any user's history" endpoint, Alembic
migrations, multi-tenant/facility scoping (requirement 11's "admin restricts
which users see which cases within a facility" — only the 2-role case/own-data
restriction is implemented), bulk account purge.

### De-identification

`healthy.data.clean.IDENTIFIER_COLUMNS` is the declared list of direct
patient-identifier columns to drop. It's empty today because neither
training source (`final_symptoms_to_disease.csv`, `Symptom2Disease.csv`)
carries a direct identifier — they're symptom/diagnosis knowledge-base rows,
not individual patient records. Any future raw source that does carry
identifiers (e.g. `Healthcare.csv`'s `Patient_ID`) should be declared here
before it's used for training.
