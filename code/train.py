import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Device (GPU if available)
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==========================
# Load Dataset
# ==========================

print("Loading dataset...")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "datasets.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

data = pd.read_csv(DATASET_PATH)
data["ner"] = data["ner"].fillna("none")

# Feature
X = data["tweet"]

# Labels
LABEL_COLUMNS = ["sentiment", "emotion", "topic", "ner"]

# ==========================
# Encode Labels
# ==========================

# Fit a LabelEncoder over the full dataset so every class has an index.
# The model itself still only ever trains on the train split.
encoders = {}
y_enc = pd.DataFrame(index=data.index)

for col in LABEL_COLUMNS:
    enc = LabelEncoder()
    y_enc[col] = enc.fit_transform(data[col])
    encoders[col] = enc
    print(f"{col:<10} classes: {enc.classes_.size}")

# ==========================
# Train/Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_enc,
    test_size=0.2,
    random_state=42,
    stratify=data["sentiment"]      # Keep sentiment balanced
)

# ==========================
# TF-IDF (feature extraction, runs on CPU)
# ==========================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(f"TF-IDF features: {X_train.shape[1]}")

# ==========================
# Model
# ==========================

class TweetClassifier(nn.Module):
    """Shared backbone + one output head per label column."""

    def __init__(self, input_dim, n_classes, hidden=256, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, n) for n in n_classes]
        )

    def forward(self, x):
        h = self.backbone(x)
        return [head(h) for head in self.heads]


def make_batches(X_csr, y_df, batch_size, shuffle, device):
    """Yield (dense x_batch, y_batch) tuples, moving them to `device`."""
    n = X_csr.shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    y = torch.tensor(y_df.to_numpy(dtype=np.int64))

    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        x_batch = torch.tensor(
            X_csr[batch_idx.numpy()].toarray(),
            dtype=torch.float32,
            device=device,
        )
        y_batch = y[batch_idx].to(device)
        yield x_batch, y_batch


# ==========================
# Training
# ==========================

print("\nTraining model on", device, "...")

EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

model = TweetClassifier(
    X_train.shape[1],
    [encoders[col].classes_.size for col in LABEL_COLUMNS],
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_batch, y_batch in make_batches(
        X_train, y_train, BATCH_SIZE, shuffle=True, device=device
    ):
        optimizer.zero_grad()

        logits = model(x_batch)

        loss = sum(
            criterion(logits[i], y_batch[:, i])
            for i in range(len(LABEL_COLUMNS))
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    print(f"Epoch {epoch}/{EPOCHS} - loss: {total_loss / n_batches:.4f}")

print("Training completed!")

# ==========================
# Evaluation
# ==========================

model.eval()
pred_columns = [[] for _ in LABEL_COLUMNS]

with torch.no_grad():
    for x_batch, _ in make_batches(
        X_test, y_test, BATCH_SIZE, shuffle=False, device=device
    ):
        logits = model(x_batch)
        for i in range(len(LABEL_COLUMNS)):
            pred_columns[i].append(
                torch.argmax(logits[i], dim=1).cpu().numpy()
            )

predictions = np.column_stack([np.concatenate(col) for col in pred_columns])

print("\n========== MODEL EVALUATION ==========\n")

for i, column in enumerate(LABEL_COLUMNS):

    print(f"----- {column.upper()} -----")

    print(
        "Accuracy:",
        accuracy_score(y_test.iloc[:, i], predictions[:, i])
    )

    if encoders[column].classes_.size <= 30:
        # Full per-class report for the small label spaces
        print(
            classification_report(
                y_test.iloc[:, i],
                predictions[:, i],
                target_names=encoders[column].classes_,
                zero_division=0,
            )
        )
    else:
        # NER has thousands of classes - aggregate by entity type instead
        y_true = encoders[column].inverse_transform(y_test.iloc[:, i].to_numpy())
        y_pred = encoders[column].inverse_transform(predictions[:, i])

        def primary_type(label):
            """Leading entity type, e.g. 'EVENT' from 'EVENT: WorldCup'."""
            return label.split(":")[0].strip().upper()

        print(f"\n  {'TYPE':<7}{'Precision':>11}{'Recall':>10}{'F1':>9}{'Accuracy':>11}")
        print("  " + "-" * 48)

        for t in ("EVENT", "ORG", "PER", "LOC"):
            # Micro-aggregate exact-match TP/FP/FN across the type's classes
            tp = fp = fn = type_match = 0

            for true, pred in zip(y_true, y_pred):
                if primary_type(true) == t:
                    if true == pred:
                        tp += 1
                        type_match += 1
                    else:
                        fn += 1
                        if primary_type(pred) == t:
                            type_match += 1
                elif primary_type(pred) == t:
                    fp += 1

            support = tp + fn
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            accuracy = type_match / support if support else 0.0

            print(
                f"  {t:<7}{precision:>11.3f}{recall:>10.3f}{f1:>9.3f}{accuracy:>11.3f}"
            )

        print("\n  Accuracy = % of each type's test tweets predicted with the correct entity type.")

# ==========================
# Save Model
# ==========================

torch.save(
    {
        "state_dict": model.state_dict(),
        "input_dim": X_train.shape[1],
        "n_classes": [encoders[col].classes_.size for col in LABEL_COLUMNS],
        "label_columns": LABEL_COLUMNS,
    },
    OUTPUT_DIR / "tweet_model.pt",
)
joblib.dump(encoders, OUTPUT_DIR / "labels.pkl")
joblib.dump(vectorizer, OUTPUT_DIR / "tfidf.pkl")

print("\nModel saved successfully!")

# ==========================
# Interactive Testing
# ==========================

print("\n========== TEST THE MODEL ==========")

while True:

    comment = input("\nEnter a tweet (type 'exit' to quit): ")

    if comment.lower() == "exit":
        print("Program ended.")
        break

    x_comment = torch.tensor(
        vectorizer.transform([comment]).toarray(),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        logits = model(x_comment)

    prediction = [
        encoders[col].inverse_transform([torch.argmax(logits[i], dim=1).item()])[0]
        for i, col in enumerate(LABEL_COLUMNS)
    ]

    print("\n========== Prediction ==========")
    print("Tweet      :", comment)
    print("Sentiment  :", prediction[0])
    print("Emotion    :", prediction[1])
    print("Topic      :", prediction[2])
    print("NER        :", prediction[3])
