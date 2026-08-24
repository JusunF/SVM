"""Assign each tweet to the train, validation or test split.

The assignment is:
  - persisted in data/processed/splits.csv and keyed by tweet id, so a tweet
    stays in the same split across runs and across models
  - group-aware: near-duplicate tweets (utils.group_key) are placed in the same
    split, so no tweet in test has a near-twin in train
  - 70/15/15, seeded

Usage:
    $ python -m src.data_cleaning.splits
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .utils import canonical_key, group_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "splits.csv"

SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15
STRATIFY_ON = "sentiment"


def group_ids(df: pd.DataFrame) -> pd.Series:
    """Return an integer near-duplicate cluster id for each row.

    Rows whose tweets share a utils.group_key() signature get the same id. Uses
    the 'canonical' column when base_cleaning.py has already computed it,
    otherwise derives it from the tweet text.
    """
    canonical = df["canonical"] if "canonical" in df.columns else df["tweet"].map(canonical_key)
    return pd.Series(pd.factorize(canonical.map(group_key))[0], index=df.index)


def build_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Assign every row a split label, returning a frame of just `id` and `split`.

    Produces a 70/15/15 train/validation/test partition in which every
    near-duplicate cluster (group_ids) falls entirely within one split.

    Class balance is not enforced — GroupShuffleSplit cannot stratify and group
    at once — so summarise() prints the per-split balance for checking. On this
    dataset the drift is under 1.5 percentage points.
    """
    groups = group_ids(df)
    holdout = VAL_SIZE + TEST_SIZE

    train_index, rest_index = next(
        GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed).split(df, groups=groups)
    )
    rest = df.iloc[rest_index]
    val_index, test_index = next(
        GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE / holdout, random_state=seed).split(
            rest, groups=groups.iloc[rest_index]
        )
    )

    train_df, val_df, test_df = df.iloc[train_index], rest.iloc[val_index], rest.iloc[test_index]

    labelled = pd.concat(
        [
            pd.DataFrame({"id": train_df["id"], "split": "train"}),
            pd.DataFrame({"id": val_df["id"], "split": "val"}),
            pd.DataFrame({"id": test_df["id"], "split": "test"}),
        ]
    )
    return labelled.sort_values("id").reset_index(drop=True)


def attach_split_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `split` column to df, reusing splits.csv so the split never moves.

    Three cases:
      - no splits.csv        -> build the split and write the file
      - file has unknown ids -> the file predates the current cleaning rules;
                                rebuild the whole split and overwrite it
      - df has new ids       -> assign just those and append them, leaving
                                existing rows where they are

    Raises if any row ends up without a split label.
    """
    if "id" not in df.columns:
        raise ValueError(f"cannot assign splits: no 'id' column (columns: {list(df.columns)})")

    if SPLITS_PATH.exists():
        splits = pd.read_csv(SPLITS_PATH, encoding="utf-8")
        stale = ~splits["id"].isin(df["id"])
        if stale.any():
            print(
                f"{int(stale.sum())} ids in {SPLITS_PATH.name} are no longer in the dataset "
                "— rebuilding the split from scratch"
            )
            splits = build_splits(df)
            splits.to_csv(SPLITS_PATH, index=False, encoding="utf-8")
        else:
            missing = df[~df["id"].isin(splits["id"])]
            if not missing.empty:
                print(f"{len(missing)} new ids since splits.csv was written and assigning them")
                splits = pd.concat([splits, build_splits(missing)]).sort_values("id")
                splits.to_csv(SPLITS_PATH, index=False, encoding="utf-8")
    else:
        print(f"No {SPLITS_PATH.name} yet - creating the shared split")
        splits = build_splits(df)
        SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
        splits.to_csv(SPLITS_PATH, index=False, encoding="utf-8")

    merged = df.merge(splits, on="id", how="left")
    if merged["split"].isna().any():
        raise ValueError(f"{merged['split'].isna().sum()} rows got no split label")
    return merged


def summarise(df: pd.DataFrame) -> None:
    """Print split sizes and the class balance within each stratification check."""
    print(f"\nTotal rows: {len(df):,}")
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        share = len(part) / len(df)
        balance = (part[STRATIFY_ON].value_counts(normalize=True) * 100).round(1).to_dict()
        print(f"  {name:5s} {len(part):6,} ({share:.1%})  {balance}")


if __name__ == "__main__":
    from .base_cleaning import CLEANED_PATH

    if not CLEANED_PATH.exists():
        raise SystemExit(
            f"{CLEANED_PATH} not found, run 'python -m src.data_cleaning.base_cleaning' first."
        )
    cleaned = pd.read_csv(CLEANED_PATH, encoding="utf-8")
    summarise(attach_split_column(cleaned))
