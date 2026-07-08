import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("datasets.csv")

X = data["comment"]
y = data["sentiment"]

vectorizer = TfidfVectorizer()

X_vector = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vector,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model...")

model = LinearSVC()
model.fit(X_train, y_train)

print("Training completed!")

prediction = model.predict(X_test)

print("\n========== MODEL EVALUATION ==========")
print("Accuracy:", accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))

joblib.dump(model, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf.pkl")

print("\nModel saved successfully!")

print("\n========== TEST THE MODEL ==========")

while True:

    comment = input("\nEnter a comment (type 'exit' to quit): ")

    if comment.lower() == "exit":
        print("Program ended.")
        break

    comment_vector = vectorizer.transform([comment])

    result = model.predict(comment_vector)

    print("Predicted Sentiment:", result[0])