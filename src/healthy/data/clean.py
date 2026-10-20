"""Data cleaning and de-identification.

Takes the output of ``ingest.py`` and produces a cleaned, de-identified,
versioned dataset under ``data/processed/`` (see ``data/README.md`` for the
versioning convention: ``dataset_v{N}.csv``).

Two cleaning paths live here, matching the two-pipeline data strategy decided
for this project (see the plan under ``.claude/plans``):

- ``clean_diagnosis_dataset`` turns ``final_symptoms_to_disease.csv`` (a
  comma-joined symptom-name string per row) into the structured
  ``disease, <symptom flags>`` shape the probability classifier trains on.
- ``clean_text_cases`` prepares ``Symptom2Disease.csv``'s free-text case
  narratives as a separate dev/eval set for the future LLM reasoning layer —
  it is never merged into the classifier's training data.

Usage:
    python -m healthy.data.clean diagnosis data/raw/final_symptoms_to_disease.csv data/processed/symptom_vocab_v1.json data/processed/dataset_v1.csv
    python -m healthy.data.clean text-cases data/raw/Symptom2Disease.csv data/processed/text_cases_v1.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from healthy.data.vocab import load_vocab

# Neither raw source used for training (``final_symptoms_to_disease.csv``,
# ``Symptom2Disease.csv``) contains a direct patient identifier (no name, MRN,
# or free-text patient ID) — they are symptom/diagnosis knowledge-base-style
# rows, not individual patient records. De-identification is therefore a
# no-op for these sources; this list exists so a future raw source that does
# carry identifiers (e.g. ``Healthcare.csv``'s ``Patient_ID``) has somewhere
# to be declared before it's used for training.
IDENTIFIER_COLUMNS: list[str] = []


def _drop_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any known direct patient identifier columns."""
    cols_present = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
    return df.drop(columns=cols_present)


def _drop_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop fully-empty rows and exact duplicate rows.

    TODO: replace/extend with real cleaning logic (imputation, type
    coercion, symptom normalization, etc.) once the dataset is known.
    """
    df = df.dropna(how="all")
    df = df.drop_duplicates()
    return df


def clean(in_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Generic clean: drop identifier columns, empty rows, and duplicates.

    ``out_path`` should follow the ``dataset_v{N}.csv`` convention documented
    in ``data/README.md``.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    df = pd.read_csv(in_path)
    df = _drop_identifiers(df)
    df = _drop_missing_and_duplicates(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df


def _vectorize_symptom_text(symptom_text: pd.Series, vocab: list[str]) -> pd.DataFrame:
    """Turn a ``"symptom one, symptom two, ..."`` column into binary flags.

    ``vocab`` entries use underscores (as in ``data.csv``'s headers);
    ``symptom_text`` uses the same names with spaces, so the vocab is
    converted to the display form only for matching, while the output
    DataFrame keeps the underscore form as column names.
    """
    display_vocab = [name.replace("_", " ") for name in vocab]
    symptom_lists = symptom_text.str.split(", ")

    mlb = MultiLabelBinarizer(classes=display_vocab)
    matrix = mlb.fit_transform(symptom_lists)

    return pd.DataFrame(matrix, columns=vocab, index=symptom_text.index)


def clean_diagnosis_dataset(
    symptom_csv_path: str | Path,
    vocab_path: str | Path,
    out_path: str | Path,
) -> pd.DataFrame:
    """Vectorize ``final_symptoms_to_disease.csv`` into the classifier's input shape.

    Produces ``disease, <symptom flags>`` columns (377 binary symptom
    columns, in the order defined by ``vocab_path``), with exact-duplicate
    rows dropped.
    """
    symptom_csv_path = Path(symptom_csv_path)
    out_path = Path(out_path)

    df = pd.read_csv(symptom_csv_path)
    vocab = load_vocab(vocab_path)

    features = _vectorize_symptom_text(df["symptom_text"], vocab)
    out_df = pd.concat([df["diseases"].rename("disease"), features], axis=1)
    out_df = out_df.drop_duplicates()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    return out_df


def clean_text_cases(in_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Prepare ``Symptom2Disease.csv`` as a free-text dev/eval set for the LLM layer.

    Drops the unnamed row-index column and normalizes to ``disease, text``.
    This output is never used to train the diagnosis probability classifier.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    df = pd.read_csv(in_path)
    df = df.rename(columns={"label": "disease"})
    df = df[["disease", "text"]]
    df = _drop_missing_and_duplicates(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df


if __name__ == "__main__":
    if len(sys.argv) == 3:
        clean(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "text-cases":
        clean_text_cases(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5 and sys.argv[1] == "diagnosis":
        clean_diagnosis_dataset(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(
            "Usage:\n"
            "  python -m healthy.data.clean <in_path> <out_path>\n"
            "  python -m healthy.data.clean diagnosis <symptom_csv_path> <vocab_json_path> <out_path>\n"
            "  python -m healthy.data.clean text-cases <in_path> <out_path>"
        )
        sys.exit(1)
