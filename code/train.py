"""
Train the TF-IDF + LinearSVC models and the NER tagger.

Run this file with:

    python -m src.models.svm.train
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

from .model import (
    DEVICE,
    GPU_AVAILABLE,
    train_models,
    train_ner_tagger,
)


def main():

    print(f"Training on: {DEVICE}")

    # Load the dataset and split it into:
    # - training data
    # - testing data
    train_df, test_df = load_and_split()

    # Convert text labels into numbers.
    #
    # For example:
    #
    # positive -> 2
    # neutral  -> 1
    # negative -> 0

    encoders = build_label_encoders(train_df)

    # Display the number of classes for each task
    for task in TASKS:
        number_of_classes = len(
            encoders[task].classes_
        )
        print(
            f"{task:<10} classes: {number_of_classes}"
        )

    # Display the NER entity types
    print(
        "NER types:",
        ", ".join(ENTITY_TYPES)
    )

    # 4. Create the TF-IDF vectorizer
    vectorizer = build_vectorizer()

    # Get the tweet text from the training dataset
    texts = train_df["tweet"]

    if GPU_AVAILABLE:
        import cudf

        # cuML expects the text data in a cuDF Series
        texts = cudf.Series(
            texts.reset_index(drop=True)
        )

    # 6. Convert tweets into TF-IDF features

    # fit_transform() does two things:
    # 1. Learn the vocabulary from the training tweets
    # 2. Convert the tweets into TF-IDF feature vectors
    train_features = vectorizer.fit_transform(
        texts
    )

    # Display the number of TF-IDF features
    print(
        f"TF-IDF features: {train_features.shape[1]}"
    )

    # 7. Convert target labels into numbers

    # Create numerical labels for each classification task.
    single_targets = encode_targets(
        train_df,
        encoders
    )

    # 8. Train the SVM models

    # Train one LinearSVC model for every task.
    models = train_models(
        train_features,
        single_targets
    )

    # 9. Prepare NER labels

    # Get all possible BIO NER labels.
    ner_labels = build_ner_labels(
        train_df
    )

    # Display the NER labels
    print(
        "NER tags:",
        ", ".join(ner_labels)
    )

    # 10. Create the NER training dataset

    # Convert the training tweets into:
    #
    # ner_X -> token feature dictionaries
    # ner_y -> correct NER tags
    ner_X, ner_y = build_ner_token_dataset(
        train_df
    )

    # Display the number of tokens used for NER training
    print(
        f"NER training tokens: {len(ner_y):,}"
    )

    # 11. Train the NER SVM

    # Train:
    #
    # DictVectorizer
    #       +
    # LinearSVC
    #
    # for token-level NER prediction.
    ner_vectorizer, ner_tagger = train_ner_tagger(
        ner_X,
        ner_y,
        ner_labels
    )

    # Add the NER model to the other SVM models
    models["ner"] = ner_tagger

    # Save the NER label list as well
    models["ner_tag_labels"] = ner_labels

    # 12. Create the model directory

    # Create the directory if it does not already exist.
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 13. Save the trained models and other artifacts

    # The artifacts are:
    #
    # 1. TF-IDF vectorizer
    # 2. Label encoders
    # 3. NER vectorizer
    # 4. All trained models
    artifacts = (
        vectorizer,
        encoders,
        ner_vectorizer,
        models,
    )

    for name, artifact in zip(
        ARTIFACTS,
        artifacts
    ):

        file_path = MODEL_DIR / name

        joblib.dump(
            artifact,
            file_path
        )

    # 14. Evaluate the trained models

    report, headlines, gold_labels, pred_labels = evaluate(
        models,
        vectorizer,
        encoders,
        ner_vectorizer,
        test_df
    )

    # Display the evaluation report
    print(report)

    # 15. Save evaluation metrics

    metrics_file = MODEL_DIR / "metrics.txt"

    metrics_file.write_text(
        report,
        encoding="utf-8"
    )

    # Save the metrics using the project's metrics system
    save_metrics(
        "svm",
        len(test_df),
        headlines,
        MODEL_DIR
    )

    # 16. Save predictions
    test_ids = list(
        test_df["id"]
    )

    save_predictions(
        MODEL_DIR,
        test_ids,
        gold_labels,
        pred_labels
    )

    print(
        f"Saved artifacts, metrics, and predictions to {MODEL_DIR}"
    )

# Run main() only when this file is executed directly
if __name__ == "__main__":
    main()
