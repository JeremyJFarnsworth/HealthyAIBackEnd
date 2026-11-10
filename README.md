# Healthy — AI Diagnostic Tool (Backend)

Healthy is an AI diagnostic assistant that lets healthcare professionals submit a
patient case (symptoms, text, images, or files) and receive a ranked list of
candidate diagnoses with calibrated probability scores and recommended treatment
plans. See [`Resouces/ProjectProposal.docx`](Resouces/ProjectProposal.docx) for
the full project proposal.

This repository holds the backend: the data pipeline, the diagnosis probability
model, calibration, the LLM reasoning layer, and a FastAPI persistence/auth
layer (role-based access and chat-history storage).

## Current status

Week 5 complete (per the proposed schedule): model training, calibration, LLM
reasoning, and persistence/role-based auth are all in place. The Next.js
frontend and website (weeks 6-8) are not yet started.

## Project structure

```
data/           raw and processed datasets (git-ignored — never commit patient data)
src/healthy/    application source:
                  data/   data pipeline (ingest, clean, vocab, resample)
                  model/  training, calibration, evaluation
                  llm/    free-text -> structured diagnosis reasoning layer
                  db/     SQLAlchemy models, engine/session
                  auth/   password hashing, JWT, role-based dependencies
                  api/    FastAPI app and routes (auth, admin, chat)
models/         trained model artifacts (git-ignored)
notebooks/      exploratory data analysis
tests/          automated tests
```

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Usage (once a dataset is available)

1. Place the raw dataset under `data/raw/`.
2. Run ingestion + cleaning to produce a versioned processed dataset under
   `data/processed/` (see [`data/README.md`](data/README.md) for the
   versioning convention).
3. Run training to produce a model artifact under `models/`.

## Running the API locally

1. Set the required environment variable (no safe default — the app won't
   import without it) and, optionally, where the database should live:

   ```powershell
   $env:JWT_SECRET_KEY = "some-dev-secret"
   $env:DATABASE_URL = "sqlite:///./healthy.db"   # optional, this is the default
   ```

   ```bash
   # macOS/Linux
   export JWT_SECRET_KEY=some-dev-secret
   export DATABASE_URL=sqlite:///./healthy.db
   ```

2. Start the server from the repo root:

   ```bash
   uvicorn healthy.api.app:app --reload --app-dir src
   ```

   Tables are created automatically on startup (`Base.metadata.create_all`) —
   no migration step is needed at this stage. Open
   [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive
   Swagger UI.

3. Try the flow:
   - `POST /auth/register` with `{"username": "...", "password": "..."}` —
     always creates a `role="user"` account.
   - `POST /auth/login` (form fields, not JSON) — returns a JWT. In `/docs`,
     use the "Authorize" button, or send `Authorization: Bearer <token>`
     manually.
   - There's no admin on a fresh database — promote one by hand:
     ```bash
     python -c "import sqlite3; c=sqlite3.connect('healthy.db'); c.execute(\"UPDATE users SET role='admin' WHERE username='yourname'\"); c.commit()"
     ```
     then log in again so the new token carries the admin role.
   - Admin-only endpoints (`GET /admin/users`, `PATCH
     /admin/users/{id}/role`, `DELETE /admin/users/{id}`) require that admin
     token.
   - `POST /chat` calls into `healthy.llm.reasoning.diagnose_from_text`,
     which needs a local LM Studio server running (`LLM_BASE_URL`/
     `LLM_MODEL` env vars, same as the CLI) plus a trained model/vocab at the
     default paths. Without those it returns a 500 — `GET /chat` and
     `GET /chat/{id}` still work fine against whatever was already persisted,
     so auth/roles/history can be exercised without LM Studio running.

See `CLAUDE.md`'s Architecture section for the design decisions behind this
layer (schema, cascade deletes, role enforcement, why there's no Alembic yet).

## Tests

```bash
pytest
```
