"""Shared symptom vocabulary.

The canonical list of 377 symptom names comes from ``data.csv``'s column
headers (minus the ``diseases`` label column). Both the diagnosis dataset
(``final_symptoms_to_disease.csv``, whose ``symptom_text`` field uses the same
symptom names with underscores replaced by spaces) and the trained model's
feature order are built against this one vocabulary, so anything that later
needs to produce a symptom vector (free-text extraction, a different raw
dataset, API input validation) stays consistent with what the model was
trained on.

Usage:
    python -m healthy.data.vocab data/raw/data.csv data/processed/symptom_vocab_v1.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def build_vocab(data_csv_path: str | Path, label_column: str = "diseases") -> list[str]:
    """Read just the header of ``data_csv_path`` and return the symptom columns."""
    header = pd.read_csv(data_csv_path, nrows=0)
    return [c for c in header.columns if c != label_column]


def save_vocab(vocab: list[str], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(vocab, indent=2))


def load_vocab(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m healthy.data.vocab <data_csv_path> <out_json_path>")
        sys.exit(1)

    save_vocab(build_vocab(sys.argv[1]), sys.argv[2])
