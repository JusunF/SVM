"""Predict SVM labels for a tweet.

The classification tasks run on text put through clean_for_svm() -- the same
pipeline that produced the training corpus, so emoji reach TF-IDF as the
tokens it was fitted on ('🔥' -> 'fire'). NER tags that same cleaned word
list directly, one tag per word, same as the tagger was trained on.

Run: ``python -m src.models.svm.predict`` or pass a tweet as arguments.
"""

import sys

import joblib
import numpy as np

from src.data_cleaning.preprocess_svm import clean_for_svm

from .config import ARTIFACTS, LEGACY_MODEL_DIR, MODEL_DIR, TASKS
from .model import GPU_AVAILABLE, ner_features_to_matrix
from .ner_features import token_features


def _artifact_dir():
    """Prefer the new data/models location, fall back to pre-refactor files."""
    if all((MODEL_DIR / name).exists() for name in ARTIFACTS):
        return MODEL_DIR
    if all((LEGACY_MODEL_DIR / name).exists() for name in ARTIFACTS):
        return LEGACY_MODEL_DIR
    raise FileNotFoundError("No SVM artifacts found. Run: python -m src.models.svm.train")


def load_model():
    directory = _artifact_dir()
    return tuple(joblib.load(directory / name) for name in ("svm_models.pkl", "tfidf.pkl", "label_encoders.pkl", "ner_vectorizer.pkl"))


def confidence_from_scores(scores):
    scores = np.asarray(scores).reshape(-1).astype(float)
    if scores.size == 1:
        probability = 1 / (1 + np.exp(-scores[0]))
        return max(probability, 1 - probability)
    shifted = scores - scores.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    return probabilities.max()


def tag_words(words: list[str], models, ner_vectorizer) -> list[tuple[str, str]]:
    """One BIO tag per word, from the SVM's own per-token tagger."""
    if not words:
        return []
    token_matrix = ner_features_to_matrix(
        ner_vectorizer, [token_features(words, i) for i in range(len(words))]
    )
    predicted_ids = models["ner"].predict(token_matrix)
    tag_labels = models["ner_tag_labels"]
    return list(zip(words, (tag_labels[i] for i in predicted_ids), strict=True))


def predict(text, models=None, vectorizer=None, encoders=None, ner_vectorizer=None):
    if models is None:
        models, vectorizer, encoders, ner_vectorizer = load_model()
    cleaned = clean_for_svm(text)  # same cleaning the training corpus went through
    texts = [cleaned]
    if GPU_AVAILABLE:
        import cudf
        texts = cudf.Series(texts)
    features = vectorizer.transform(texts)
    result = {"cleaned": cleaned}
    for task in TASKS:
        predicted = models[task].predict(features)[0]
        result[task] = encoders[task].inverse_transform([predicted])[0]
        result[f"{task}_confidence"] = confidence_from_scores(models[task].decision_function(features))
    result["ner"] = tag_words(cleaned.split(), models, ner_vectorizer)
    return result


def main():
    models, vectorizer, encoders, ner_vectorizer = load_model()
    if len(sys.argv) > 1:
        tweets = [" ".join(sys.argv[1:])]
    else:
        tweets = None
    while True:
        tweet = tweets.pop() if tweets else input("Enter a tweet (empty to quit): ").strip()
        if not tweet:
            break
        result = predict(tweet, models, vectorizer, encoders, ner_vectorizer)
        for task in TASKS:
            print(f"{task.title():9s}: {result[task]} ({result[f'{task}_confidence']:.0%} sure)")
        tagged = " ".join(f"{w}[{t}]" if t != "O" else w for w, t in result["ner"])
        print(f"NER      : {tagged}")
        if tweets is not None:
            break


if __name__ == "__main__":
    main()
