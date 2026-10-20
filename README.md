# Healthy — AI Diagnostic Tool (Backend)

Healthy is an AI diagnostic assistant that lets healthcare professionals submit a
patient case (symptoms, text, images, or files) and receive a ranked list of
candidate diagnoses with calibrated probability scores and recommended treatment
plans. See [`Resouces/ProjectProposal.docx`](Resouces/ProjectProposal.docx) for
the full project proposal.

This repository holds the backend: the data pipeline, the diagnosis probability
model, and (in later weeks) the API, calibration, and LLM reasoning layer.

## Current status

Week 1–2 focus (per the proposed schedule): find a dataset, clean/de-identify
it, and train an initial diagnosis probability model. The structure below is
scaffolding for that work — most scripts are stubs until a real dataset is in
hand.

## Project structure

```
data/           raw and processed datasets (git-ignored — never commit patient data)
src/healthy/    application source (data pipeline, model training)
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

## Tests

```bash
pytest
```
