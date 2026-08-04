import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report

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
y = data[["sentiment", "emotion", "topic", "ner"]]

# ==========================
# Train/Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y["sentiment"]      # Keep sentiment balanced
)

# ==========================
# TF-IDF
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

# ==========================
# Train Model
# ==========================

print("Training model...")

model = MultiOutputClassifier(LinearSVC())

model.fit(X_train, y_train)

print("Training completed!")

# ==========================
# Evaluation
# ==========================

prediction = model.predict(X_test)

columns = ["sentiment", "emotion", "topic","NER"]

print("\n========== MODEL EVALUATION ==========\n")

for i, column in enumerate(columns):

    print(f"----- {column.upper()} -----")

    print(
        "Accuracy:",
        accuracy_score(y_test.iloc[:, i], prediction[:, i])
    )

    print(
        classification_report(
            y_test.iloc[:, i],
            prediction[:, i]
        )
    )

# ==========================
# Save Model
# ==========================

joblib.dump(model, OUTPUT_DIR / "svm_model.pkl")
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

    comment_vector = vectorizer.transform([comment])

    result = model.predict(comment_vector)

    sentiment = result[0][0]
    emotion = result[0][1]
    topic = result[0][2]
    ner = result[0][3]

    print("\n========== Prediction ==========")
    print("Tweet      :", comment)
    print("Sentiment  :", sentiment)
    print("Emotion    :", emotion)
    print("Topic      :", topic)
    print("NER        :", ner)