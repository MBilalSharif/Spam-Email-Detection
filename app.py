from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load pickled TF-IDF and model
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model/spam_detector_model.pkl", "rb") as f:
    model = pickle.load(f)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form["email"]  # get text from form

    # Vectorize input using TF-IDF
    vectorized = tfidf.transform([email_text])

    # Make prediction
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0] if hasattr(model, 'predict_proba') else None

    label = "SPAM" if prediction == 1 else "HAM"

    # Return result to template
    return render_template(
        "index.html",
        prediction_text=f"Prediction: {label}",
        confidence=f"Ham: {probability[0]*100:.2f}% | Spam: {probability[1]*100:.2f}%" if probability is not None else ""
    )

if __name__ == "__main__":
    app.run(debug=True)