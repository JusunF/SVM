import joblib
import spacy
import torch
import torch.nn as nn
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

# Load trained model, label encoders and TF-IDF
checkpoint = torch.load(MODEL_DIR / "tweet_model.pt", map_location=device)
encoders = joblib.load(MODEL_DIR / "labels.pkl")
vectorizer = joblib.load(MODEL_DIR / "tfidf.pkl")
label_columns = checkpoint["label_columns"]


class TweetClassifier(nn.Module):
    """Must mirror the architecture used in train.py."""

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


model = TweetClassifier(
    checkpoint["input_dim"], checkpoint["n_classes"]
).to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()


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

    x_comment = torch.tensor(
        vectorizer.transform([comment]).toarray(),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        logits = model(x_comment)

    prediction = [
        encoders[col].inverse_transform(
            [torch.argmax(logits[i], dim=1).item()]
        )[0]
        for i, col in enumerate(label_columns)
    ]

    sentiment = prediction[0]
    emotion = prediction[1]
    topic = prediction[2]
    predicted_ner = prediction[3]

    ner = extract_ner(comment)

    print("\n========== RESULT ==========")
    print("Tweet      :", comment)
    print("Sentiment  :", sentiment)
    print("Emotion    :", emotion)
    print("Topic      :", topic)
    print("NER (model) :", predicted_ner)
    print("NER (spaCy) :", ner)
