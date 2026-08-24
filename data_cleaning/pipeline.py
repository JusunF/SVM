"""Shared machinery for running a model-specific cleaning pipeline.

Each preprocess_*.py defines a STEPS list of atomic helpers from utils.py;
this module runs those steps over the merged dataset and writes the
model-ready CSV to data/processed/.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .base_cleaning import CLEANED_PATH, PROCESSED_DIR

Step = Callable[[str], str]


def run_steps(text: str, steps: list[Step]) -> str:
    for step in steps:
        text = step(text)
    return text


def preprocess_dataset(steps: list[Step], out_name: str, cleaned_path: Path = CLEANED_PATH) -> pd.DataFrame:
    """Apply a step chain to cleaned_tweets.csv and save the result.

    Rewrites the 'tweet' column and leaves every other column, and the row
    count, untouched — the output has the same ids and the same `split` labels
    as the input, which is what lets the three models be compared on one test
    set. Raises AssertionError if a step chain changes the row count; row
    filtering belongs in base_cleaning.py.

    Warns, but does not drop, when a step chain empties a tweet's text.
    """
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"{cleaned_path} not found — run 'python -m src.data_cleaning.base_cleaning' first."
        )

    df = pd.read_csv(cleaned_path, encoding="utf-8")
    df["tweet"] = df["tweet"].fillna("").astype(str)

    before = len(df)
    df["tweet"] = df["tweet"].map(lambda t: run_steps(t, steps))
    if len(df) != before:
        raise AssertionError(
            f"{out_name}: preprocessing changed the row count ({before} -> {len(df)}). "
            "Row filtering belongs in base_cleaning.py, not here."
        )

    empty = int((df["tweet"].str.len() == 0).sum())
    if empty:
        print(f"{out_name}: warning — {empty} rows cleaned to empty text (kept, to hold the row set)")
    print(f"{out_name}: {len(df)} rows (unchanged, as required)")

    out_path = PROCESSED_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved to {out_path}")
    return df
