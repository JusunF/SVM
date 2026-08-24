"""Build cleaned_tweets.csv, the single row set all three models train on.

Reads merged_tweets.csv and, in order:
  1. repairs mojibake, decodes HTML entities, strips 'RT @user:' prefixes,
     normalizes whitespace
  2. drops rows that are empty once normalized, and deduplicates on
     utils.canonical_key()
  3. keeps only lang == 'en'
  4. stamps each row with its train/val/test `split` (splits.py)

Steps 2 and 3 are the only places in the project that change the row count, and
they run before step 4. Each model's preprocess_*.py then rewrites the text
without adding or removing rows, so all three models share one row set and one
split.

Run:  python -m src.data_cleaning.base_cleaning
"""

from pathlib import Path

import pandas as pd

from .splits import attach_split_column, summarise
from .utils import (
    canonical_key,
    fix_mojibake,
    normalize_whitespace,
    remove_rt_prefix,
    unescape_html,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MERGED_PATH = PROCESSED_DIR / "merged_tweets.csv"
CLEANED_PATH = PROCESSED_DIR / "cleaned_tweets.csv"


def structural_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize tweet text, then drop empty and duplicate rows.

    Text changes are limited to repairing mojibake from the scrape
    ('didnâ€™t' -> "didn’t"), decoding HTML entities ('&amp;' -> '&'),
    stripping the 'RT @user:' retweet prefix, and collapsing whitespace, so a
    retweet and its original become the same string.

    Mojibake is repaired first, so canonical_key sees the real characters. It
    does not change which rows survive: canonical_key strips non-alphanumerics
    anyway, so the repair is verified to leave the row set byte-identical.

    Duplicates are judged by utils.canonical_key(), which ignores case,
    punctuation, links and mentions — a stricter test than comparing the tweet
    text itself. Adds a 'canonical' column holding that key, which splits.py
    reuses to group near-duplicates.
    """
    df = df.copy()
    df["tweet"] = df["tweet"].fillna("").astype(str)
    df["tweet"] = (
        df["tweet"]
        .map(fix_mojibake)
        .map(unescape_html)
        .map(remove_rt_prefix)
        .map(normalize_whitespace)
    )

    df["canonical"] = df["tweet"].map(canonical_key)
    df = df[df["canonical"].str.len() > 0]
    df = df.drop_duplicates(subset="canonical", keep="first")
    return df.reset_index(drop=True)


def build_cleaned_dataset(merged_path: Path = MERGED_PATH, out_path: Path = CLEANED_PATH) -> pd.DataFrame:
    if not merged_path.exists():
        raise FileNotFoundError(f"{merged_path} not found — run 'python scripts/merge_datasets.py' first.")

    df = pd.read_csv(merged_path, encoding="utf-8")
    if "tweet" not in df.columns:
        raise ValueError(f"{merged_path} has no 'tweet' column (columns: {list(df.columns)})")

    before = len(df)
    df = structural_clean(df)
    print(f"Cleaned {before} rows -> {len(df)} after removing empties/duplicates")

    before_lang = len(df)
    df = df[df["lang"] == "en"].reset_index(drop=True)
    print(f"Filtered {before_lang} rows -> {len(df)} keeping lang == 'en'")

    # 'canonical' is dropped after the split is assigned — it identifies rows,
    # and is not a feature.
    df = attach_split_column(df).drop(columns="canonical")
    summarise(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig adds a BOM so Excel displays emoji/non-ASCII correctly
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved cleaned dataset to {out_path}")
    return df


if __name__ == "__main__":
    build_cleaned_dataset()
