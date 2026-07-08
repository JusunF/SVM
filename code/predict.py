import joblib

model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

comment = input("Enter comment: ")

comment_vector = vectorizer.transform([comment])

prediction = model.predict(comment_vector)

print("Sentiment:", prediction[0])  