import pickle
from pathlib import Path

from flask import Flask, jsonify, request

MODEL_PATH = Path("model.pkl")

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "age" not in data or "height" not in data:
        return jsonify({"error": "Request body must be JSON with 'age' and 'height' fields."}), 400

    try:
        age = float(data["age"])
        height = float(data["height"])
    except (TypeError, ValueError):
        return jsonify({"error": "'age' and 'height' must be numeric."}), 400

    prediction = model.predict([[age, height]])
    return jsonify({"eu_shoe_size": round(float(prediction[0]), 1)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
