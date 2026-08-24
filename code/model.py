"""SVM backend selection and model construction."""

import numpy as np

from .config import TASKS

try:
    import cupy as cp
    from cuml.feature_extraction.text import TfidfVectorizer
    from cuml.svm import LinearSVC
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    GPU_AVAILABLE = False

if not GPU_AVAILABLE:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

DEVICE = "cuda" if GPU_AVAILABLE else "cpu"


def make_vectorizer(kwargs):
    return TfidfVectorizer(**kwargs)


def to_numpy(values):
    return cp.asnumpy(values) if GPU_AVAILABLE else np.asarray(values)


WEIGHT_CAP = 10.0  # same cap the BiLSTM and RoBERTa-CNN losses apply


def class_weights(target) -> dict[int, float]:
    """Inverse-frequency weight per class: min(n / (k * count), WEIGHT_CAP).

    n is the number of samples and k the number of classes present, so a class
    appearing less often than average weighs more than 1. The cap keeps a very
    rare class from dominating the objective, and matches the weighting the
    BiLSTM and RoBERTa-CNN build into their CrossEntropyLoss.
    """
    counts = np.bincount(np.asarray(target))
    n, k = len(target), (counts > 0).sum()
    return {
        i: min(n / (k * count), WEIGHT_CAP) for i, count in enumerate(counts) if count > 0
    }


def train_models(features, single_targets):
    """Fit one multiclass SVM per task.

    Class-weighted with class_weights(). cuML's LinearSVC takes no
    class_weight argument, so on GPU the same weights are applied per row
    through sample_weight instead; both routes optimise the same objective.
    """
    models = {}
    for task in TASKS:
        target = single_targets[task]
        weights = class_weights(target)
        sample_weight = np.array([weights[int(value)] for value in target], dtype=np.float32)
        if GPU_AVAILABLE:
            models[task] = LinearSVC().fit(
                features, cp.asarray(target), sample_weight=cp.asarray(sample_weight)
            )
        else:
            models[task] = LinearSVC(class_weight=weights).fit(features, target)
    return models


def ner_features_to_matrix(ner_vectorizer, token_dicts: list[dict]):
    """DictVectorizer -> sparse matrix, downcast to int32 indices.

    DictVectorizer can emit int64-indexed sparse matrices depending on the
    scipy build; sklearn's LinearSVC sparse fit/predict path requires 32-bit
    indices and raises otherwise. Every caller feeding the NER tagger (train
    and predict) goes through this, not just fit time.
    """
    matrix = ner_vectorizer.transform(token_dicts).tocsr()
    matrix.indices = matrix.indices.astype(np.int32)
    matrix.indptr = matrix.indptr.astype(np.int32)
    return matrix


def train_ner_tagger(X_dicts: list[dict], y_tags: list[str], tag_labels: list[str]):
    """Train a per-token BIO tagger: DictVectorizer + a class-weighted LinearSVC.

    Always CPU (sklearn), regardless of GPU_AVAILABLE - cuML has no
    DictVectorizer equivalent, and per-token feature matrices are small
    enough that this isn't a bottleneck either way.

    tag_labels fixes the tag -> integer mapping so predict-time decoding
    matches training exactly, independent of which tags happen to appear in
    this particular split.
    """
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.svm import LinearSVC as CPULinearSVC

    tag_to_id = {tag: i for i, tag in enumerate(tag_labels)}
    target = np.array([tag_to_id[tag] for tag in y_tags])

    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit(X_dicts)
    features = ner_features_to_matrix(vectorizer, X_dicts)

    weights = class_weights(target)
    tagger = CPULinearSVC(class_weight=weights).fit(features, target)
    return vectorizer, tagger
