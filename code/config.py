"""Settings for the TF-IDF + LinearSVC baseline."""

from pathlib import Path

# re-exported so the one-vs-rest column order matches the order the shared
# entity-presence metric binarises with
from src.models.metrics import ENTITY_TYPES  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "svm_input.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "svm"
# Old training runs saved artifacts next to this source package. This fallback
# lets predict.py use them until the reorganised trainer is run once.
LEGACY_MODEL_DIR = Path(__file__).resolve().parent

SEED = 42  # train/val/test assignment lives in data/processed/splits.csv
TASKS = ["sentiment", "emotion", "topic"]
TFIDF_KWARGS = {"lowercase": True, "stop_words": "english", "ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "sublinear_tf": True}
ARTIFACTS = ("tfidf.pkl", "label_encoders.pkl", "ner_vectorizer.pkl", "svm_models.pkl")
