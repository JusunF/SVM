"""Dataset loading, splitting, label encoding, and TF-IDF feature creation."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import DATA_PATH, TASKS, TFIDF_KWARGS
from .model import make_vectorizer
from .ner_bio import add_bio_tags
from .ner_features import token_features


def load_and_split():
    """Load the cleaned SVM input and return the shared train/test split.

    Split labels come from data/processed/splits.csv (70/15/15). The 'val'
    slice is unused - LinearSVC has no early stopping.
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df[df["lang"] == "en"]
    df = df.dropna(subset=["tweet", *TASKS]).reset_index(drop=True)
    df["ner"] = df["ner"].fillna("none")

    if "split" not in df.columns:
        raise ValueError(
            f"{DATA_PATH} has no 'split' column - re-run "
            "'python -m src.data_cleaning.base_cleaning' then "
            "'python -m src.data_cleaning.preprocess_svm'"
        )

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Split: train {len(train_df)}, test {len(test_df)} (val slice unused)")
    return train_df, test_df


def build_label_encoders(train_df) -> dict[str, LabelEncoder]:
    return {task: LabelEncoder().fit(train_df[task].astype(str)) for task in TASKS}


def encode_targets(df, encoders) -> dict:
    """One integer label array per task in TASKS."""
    return {task: encoders[task].transform(df[task].astype(str)) for task in TASKS}


def build_vectorizer():
    return make_vectorizer(TFIDF_KWARGS)


def build_ner_labels(train_df) -> list[str]:
    """The sorted BIO tag vocabulary (position = class number), from the training split."""
    tagged = add_bio_tags(train_df)
    return sorted({tag for tags in tagged["bio_tags"] for tag in tags.split()})


def build_ner_token_dataset(df) -> tuple[list[dict], list[str]]:
    """Flatten every tweet into (per-token features, gold BIO tag) pairs.

    One row per word across the whole dataframe, not one row per tweet -
    this is what the per-token tagger actually trains and is scored on.
    """
    tagged = add_bio_tags(df)
    features: list[dict] = []
    tags: list[str] = []
    for text, bio_tags in zip(tagged["tweet"], tagged["bio_tags"], strict=True):
        words = str(text).split()
        word_tags = bio_tags.split()
        features.extend(token_features(words, i) for i in range(len(words)))
        tags.extend(word_tags)
    return features, tags
