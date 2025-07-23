from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob

# Load LSTM model and tokenizer
model = load_model("lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200  # Must match your training setting

# Bias keywords list (you can expand it later)
bias_keywords = ["hoax", "exposed", "shocking", "reveal", "propaganda", "agenda", "secret", "hidden"]

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def check_bias_words(text):
    return [word for word in bias_keywords if word.lower() in text.lower()]

# Flask setup
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        # Preprocessing
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=max_len)

        # LSTM Prediction
        prediction = model.predict(padded)[0][0]
        label = "REAL" if prediction >= 0.5 else "FAKE"

        # Add-on Features
        sentiment = get_sentiment(text)
        bias_words = check_bias_words(text)

        return jsonify({
            "label": label,
            "confidence": round(float(prediction), 2),
            "sentiment": sentiment,
            "bias_words": bias_words
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
