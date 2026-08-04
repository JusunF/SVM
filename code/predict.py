import joblib
import spacy
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

# Load trained SVM and TF-IDF
model = joblib.load(MODEL_DIR / "svm_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "tfidf.pkl")


def extract_ner(text):
    doc = nlp(text)

    persons = []
    orgs = []
    locs = []
    events = []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            persons.append(ent.text)

        elif ent.label_ == "ORG":
            orgs.append(ent.text)

        elif ent.label_ in ["GPE", "LOC"]:
            locs.append(ent.text)

        elif ent.label_ == "EVENT":
            events.append(ent.text)

    result = []

    if persons:
        result.append("PER: " + ", ".join(sorted(set(persons))))

    if orgs:
        result.append("ORG: " + ", ".join(sorted(set(orgs))))

    if locs:
        result.append("LOC: " + ", ".join(sorted(set(locs))))

    if events:
        result.append("EVENT: " + ", ".join(sorted(set(events))))

    if not result:
        return "None"

    return " | ".join(result)


print("========== FIFA World Cup Tweet Classifier ==========")

while True:

    comment = input("\nEnter a tweet (type 'exit' to quit): ")

    if comment.lower() == "exit":
        break

    comment_vector = vectorizer.transform([comment])

    prediction = model.predict(comment_vector)

    sentiment = prediction[0][0]
    emotion = prediction[0][1]
    topic = prediction[0][2]
    predicted_ner = prediction[0][3]

    ner = extract_ner(comment)

    print("\n========== RESULT ==========")
    print("Tweet      :", comment)
    print("Sentiment  :", sentiment)
    print("Emotion    :", emotion)
    print("Topic      :", topic)
    print("NER (model) :", predicted_ner)
    print("NER (spaCy) :", ner)
