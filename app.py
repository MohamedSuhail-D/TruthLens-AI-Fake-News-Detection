from flask import Flask, render_template, request, jsonify
import joblib
import re
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

model = joblib.load("models/fake_news_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

feature_names = tfidf.get_feature_names_out()
coefficients = model.coef_[0]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_news(text):
    cleaned_text = clean_text(text)

    if not cleaned_text:
        return "Invalid input", 0.0, [], []

    text_tfidf = tfidf.transform([cleaned_text])

    prediction = model.predict(text_tfidf)[0]

    probabilities = model.predict_proba(text_tfidf)[0]

    confidence = probabilities[prediction] * 100

    feature_values = text_tfidf.toarray()[0] * coefficients

    real_features = []
    fake_features = []
    excluded_words = ENGLISH_STOP_WORDS.union({"said", "video", "via"})

    for i in feature_values.argsort()[::-1]:
        if feature_values[i] <= 0:
            break

        word = feature_names[i]

        if any(part in excluded_words or len(part) <= 2 for part in word.split()):
            continue

        real_features.append({
            "word": word,
            "weight": round(float(feature_values[i]), 4)
        })

        if len(real_features) == 5:
            break

    for i in feature_values.argsort():
        if feature_values[i] >= 0:
            break

        word = feature_names[i]

        if any(part in excluded_words or len(part) <= 2 for part in word.split()):
            continue

        fake_features.append({
            "word": word,
            "weight": round(float(feature_values[i]), 4)
        })

        if len(fake_features) == 5:
            break

    if prediction == 0:
        result = "Likely Fake"
    else:
        result = "Likely Real"

    return result, confidence, real_features, fake_features


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    text = data.get("text", "")

    result, confidence, real_features, fake_features = predict_news(text)

    if result == "Invalid input":
        return jsonify({
            "error": "Please enter a news article or headline."
        }), 400

    return jsonify({
    "prediction": result,
    "confidence": round(confidence, 2),
    "real_features": real_features,
    "fake_features": fake_features
})
@app.route("/feature-importance")
def feature_importance():
    real_features = []
    fake_features = []
    excluded_words = ENGLISH_STOP_WORDS.union({"said", "video", "via"})

    for i in coefficients.argsort()[::-1]:
        word = feature_names[i]
        if any(part in excluded_words or len(part) <= 2 for part in word.split()):
            continue

        real_features.append({
            "word": word,
            "weight": round(float(coefficients[i]), 4)
        })

        if len(real_features) == 5:
            break

    for i in coefficients.argsort():
        word = feature_names[i]

        if any(part in excluded_words or len(part) <= 2 for part in word.split()):
            continue
        fake_features.append({
            "word": word,
            "weight": round(float(coefficients[i]), 4)
        })

        if len(fake_features) == 5:
            break

    return jsonify({
        "real": real_features,
        "fake": fake_features
    })

if __name__ == "__main__":
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))