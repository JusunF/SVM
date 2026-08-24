"""Train the TF-IDF + LinearSVC baseline, plus its own per-token NER tagger.

Run: ``python -m src.models.svm.train``
"""

import joblib

from src.models.metrics import save_metrics, save_predictions

from .config import ARTIFACTS, ENTITY_TYPES, MODEL_DIR, TASKS
from .data import (
    build_label_encoders,
    build_ner_labels,
    build_ner_token_dataset,
    build_vectorizer,
    encode_targets,
    load_and_split,
)
from .evaluate import evaluate
from .model import DEVICE, GPU_AVAILABLE, train_models, train_ner_tagger


def main():
    print(f"Training on: {DEVICE}")
    train_df, test_df = load_and_split()
    encoders = build_label_encoders(train_df)
    for task in TASKS:
        print(f"{task:<10} classes: {len(encoders[task].classes_)}")
    print("NER types:", ", ".join(ENTITY_TYPES))

    vectorizer = build_vectorizer()
    texts = train_df["tweet"]
    if GPU_AVAILABLE:
        import cudf
        texts = cudf.Series(texts.reset_index(drop=True))
    train_features = vectorizer.fit_transform(texts)
    print(f"TF-IDF features: {train_features.shape[1]}")
    single_targets = encode_targets(train_df, encoders)
    models = train_models(train_features, single_targets)

    ner_labels = build_ner_labels(train_df)
    print("NER tags:", ", ".join(ner_labels))
    ner_X, ner_y = build_ner_token_dataset(train_df)
    print(f"NER training tokens: {len(ner_y):,}")
    ner_vectorizer, ner_tagger = train_ner_tagger(ner_X, ner_y, ner_labels)
    models["ner"] = ner_tagger
    models["ner_tag_labels"] = ner_labels

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, artifact in zip(ARTIFACTS, (vectorizer, encoders, ner_vectorizer, models)):
        joblib.dump(artifact, MODEL_DIR / name)
    report, headlines, gold_labels, pred_labels = evaluate(
        models, vectorizer, encoders, ner_vectorizer, test_df
    )
    print(report)
    (MODEL_DIR / "metrics.txt").write_text(report, encoding="utf-8")
    save_metrics("svm", len(test_df), headlines, MODEL_DIR)
    save_predictions(MODEL_DIR, list(test_df["id"]), gold_labels, pred_labels)
    print(f"Saved artifacts, metrics, and predictions to {MODEL_DIR}")


if __name__ == "__main__":
    main()
