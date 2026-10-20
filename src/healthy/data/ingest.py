"""Data ingestion.

Loads a raw dataset from ``data/raw/`` and returns it as a pandas DataFrame,
with no cleaning or de-identification applied yet (see ``clean.py`` for that
step). This is intentionally generic until the real dataset's format is
known — update ``_load_raw`` once the downloaded dataset's format (CSV,
JSON, Excel, etc.) is confirmed.

Usage:
    python -m healthy.data.ingest data/raw/<dataset_file> data/processed/ingested_v1.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _load_raw(raw_path: Path) -> pd.DataFrame:
    """Load the raw dataset file into a DataFrame.

    TODO: update once the real dataset format is known. Currently assumes
    CSV; extend with a suffix check (``.json``, ``.xlsx``, etc.) as needed.
    """
    if raw_path.suffix.lower() == ".csv":
        return pd.read_csv(raw_path)
    if raw_path.suffix.lower() == ".json":
        return pd.read_json(raw_path)
    raise ValueError(f"Unsupported raw dataset format: {raw_path.suffix}")


def ingest(raw_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Load the raw dataset and write an unmodified copy to ``out_path``.

    This is the reproducible entry point referenced by the Data Ingestion
    requirement: given the same ``raw_path``, this always produces the same
    output. Cleaning / de-identification happens in ``clean.py``.
    """
    raw_path = Path(raw_path)
    out_path = Path(out_path)

    df = _load_raw(raw_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m healthy.data.ingest <raw_path> <out_path>")
        sys.exit(1)

    ingest(sys.argv[1], sys.argv[2])
