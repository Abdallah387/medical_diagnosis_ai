from flask import Flask, request, jsonify
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

MODEL_PATH = "medical_model"
device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

with open(f"{MODEL_PATH}/labels.json") as f:
    label_map = json.load(f)

id2label = {int(k): v for k, v in label_map.items()}


def clean_symptoms(text):
    text = re.sub(
        r"The patient experiences?|Patient (has|reports?|experiences?)",
        "", text, flags=re.IGNORECASE
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def predict_disease(symptoms):
    symptoms = clean_symptoms(symptoms)
    inputs = tokenizer(
        symptoms,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64        # ✅ matches training
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
        top3 = torch.topk(probs, 3)

    return {
        "symptoms": symptoms,
        "top_predictions": [
            {
                "disease": id2label[int(top3.indices[i])],
                "probability": float(top3.values[i])
            }
            for i in range(3)
        ]
    }


@app.route("/")
def home():
    return "Medical Diagnosis AI API is running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", "")
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400
    return jsonify(predict_disease(symptoms))


if __name__ == "__main__":
    app.run(debug=True)
