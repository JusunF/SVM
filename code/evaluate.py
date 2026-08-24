"""Score the trained SVM artifacts on the test split.

Prints a report and writes it to metrics.txt, plus metrics.json for
scripts/compare_models.py. The sentiment/emotion/topic metrics come from
src/models/metrics.py, shared with the other two models; NER runs the
per-token tagger over each test tweet's own words, then collapses the
predicted tags to entity-type presence via types_from_bio() the same way
BiLSTM/RoBERTa-CNN's evaluators do, so all three land on one shared metric.

Run:  python -m src.models.svm.evaluate
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from src.models.metrics import (
    entity_presence_report,
    findable_entity_types,
    save_metrics,
    save_predictions,
    single_label_report,
    types_from_bio,
)

from .config import MODEL_DIR, TASKS
from .data import encode_targets, load_and_split
from .model import GPU_AVAILABLE, ner_features_to_matrix, to_numpy
from .ner_bio import add_bio_tags
from .ner_features import token_features
from .predict import load_model


def evaluate(models, vectorizer, encoders, ner_vectorizer, test_df):
    texts = test_df["tweet"]
    if GPU_AVAILABLE:
        import cudf
        texts = cudf.Series(texts.reset_index(drop=True))
    features = vectorizer.transform(texts)
    single_targets = encode_targets(test_df, encoders)

    lines = ["========== MODEL EVALUATION (SVM) ==========", f"Test rows: {len(test_df)}"]
    headlines = {}
    gold_labels: dict[str, list[str]] = {}
    pred_labels: dict[str, list[str]] = {}

    for task in TASKS:
        predictions = to_numpy(models[task].predict(features))
        report, headline = single_label_report(
            single_targets[task], predictions, list(encoders[task].classes_)
        )
        headlines[task] = headline
        gold_labels[task] = list(encoders[task].inverse_transform(single_targets[task]))
        pred_labels[task] = list(encoders[task].inverse_transform(predictions))
        lines.extend([
            f"\n===== {task.upper()} =====",
            (
                f"Accuracy: {headline['accuracy']:.4f}  "
                f"macro P {headline['macro_precision']:.4f}  "
                f"R {headline['macro_recall']:.4f}  F1 {headline['macro_f1']:.4f}"
            ),
            report,
        ])

    ner_headline, ner_lines = _evaluate_ner(models, ner_vectorizer, test_df)
    headlines["ner"] = ner_headline
    lines.extend(ner_lines)

    return "\n".join(lines), headlines, gold_labels, pred_labels


def _evaluate_ner(models, ner_vectorizer, test_df):
    """Tag every test tweet's own words, then score two ways: token-level BIO
    (this model's own diagnostic) and entity-type presence (the framing all
    three models share, via types_from_bio())."""
    ner_labels = models["ner_tag_labels"]
    tag_to_id = {tag: i for i, tag in enumerate(ner_labels)}
    tagged = add_bio_tags(test_df)

    true_tag_ids, pred_tag_ids = [], []
    predicted_types, true_types = [], []

    for text, gold_bio, ner_raw in zip(
        tagged["tweet"], tagged["bio_tags"], test_df["ner"], strict=True
    ):
        words = str(text).split()
        token_matrix = ner_features_to_matrix(
            ner_vectorizer, [token_features(words, i) for i in range(len(words))]
        )
        predicted_ids = list(models["ner"].predict(token_matrix))
        predicted_tags = [ner_labels[i] for i in predicted_ids]

        true_tag_ids.extend(tag_to_id[tag] for tag in gold_bio.split())
        pred_tag_ids.extend(predicted_ids)
        predicted_types.append(types_from_bio(predicted_tags))
        true_types.append(findable_entity_types(text, ner_raw))

    presence_report, headline = entity_presence_report(true_types, predicted_types)
    lines = ["\n===== NER =====", presence_report]

    # token accuracy here counts the dominant 'O' class, so the entity-only
    # line below it is the meaningful figure
    lines.append("\n--- token-level BIO detail (this model only) ---")
    lines.append(f"Token accuracy incl. 'O': {accuracy_score(true_tag_ids, pred_tag_ids):.4f}")
    lines.append(
        classification_report(
            true_tag_ids, pred_tag_ids, labels=range(len(ner_labels)),
            target_names=ner_labels, digits=4, zero_division=0,
        )
    )
    entity_tag_ids = [i for i, name in enumerate(ner_labels) if name != "O"]
    p, r, f1, _ = precision_recall_fscore_support(
        true_tag_ids, pred_tag_ids, labels=entity_tag_ids, average="macro", zero_division=0
    )
    lines.append(f"Entity tags only (excluding O): precision {p:.4f}, recall {r:.4f}, F1 {f1:.4f}")

    return headline, lines


def main():
    models, vectorizer, encoders, ner_vectorizer = load_model()
    _, test_df = load_and_split()
    report, headlines, gold_labels, pred_labels = evaluate(
        models, vectorizer, encoders, ner_vectorizer, test_df
    )
    print(report)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "metrics.txt").write_text(report, encoding="utf-8")
    save_metrics("svm", len(test_df), headlines, MODEL_DIR)
    save_predictions(MODEL_DIR, list(test_df["id"]), gold_labels, pred_labels)
    print(f"\nSaved to {MODEL_DIR / 'metrics.txt'}, metrics.json, and predictions.csv")


if __name__ == "__main__":
    main()
