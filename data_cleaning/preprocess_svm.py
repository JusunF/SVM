"""Minimal cleaning for SVM (TF-IDF + LinearSVC).

TF-IDF works on word n-grams, so we:
  - remove links entirely (URLs add noise, no signal for sentiment)
  - convert emoji to single underscore-joined tokens (e.g. '❤️' -> ' red_heart ')
    so they survive TF-IDF vectorization as distinct features
  - lowercase + remove punctuation + normalize whitespace (standard TF-IDF prep)

Run:  python -m src.data_cleaning.preprocess_svm
"""

from .pipeline import preprocess_dataset, run_steps
from .utils import (
    demojize_to_token,
    lowercase,
    normalize_whitespace,
    remove_links,
    remove_punctuation,
    remove_unicode_punctuation,
)

STEPS = [
    remove_links,
    # unicode-aware, before demojize: strips the curly quotes Twitter inserts,
    # which the ASCII-only remove_punctuation below leaves behind and which
    # otherwise split the vocabulary ('its' and 'it’s' as separate features).
    remove_unicode_punctuation,
    demojize_to_token,
    lowercase,
    # ASCII pass, after demojize: strips the '_' in 'soccer_ball' -> 'soccerball'
    remove_punctuation,
    normalize_whitespace,
]


def clean_for_svm(text: str, language: str = "en") -> str:
    return run_steps(text, STEPS)


if __name__ == "__main__":
    preprocess_dataset(STEPS, "svm_input.csv")
