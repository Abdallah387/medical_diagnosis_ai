from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

app = Flask(__name__)

MODEL_PATH = "E:\project\medical_diagnosis_ai\medical_model"
device = torch.device("cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Load labels
import json

# Load labels file
with open("medical_model/labels.json") as f:
    label_map = json.load(f)

# تحويله لعكسه
id2label = {i: condition for condition, i in label_map.items()}
def predict_disease(symptoms):
    # Tokenization with same settings as training
    inputs = tokenizer(
        symptoms,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        top3 = torch.topk(probs, 3)

        results = []
        for i in range(3):
            idx = int(top3.indices[i])
            prob = float(top3.values[i])
            # Convert idx to string if labels.json has string keys
            results.append({
                "disease": labels[str(idx)],
                "probability": prob
            })
        return {
            "symptoms": symptoms,
            "top_predictions": results
        }

@app.route("/")
def home():
    return "Medical Diagnosis AI API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data["symptoms"]
    result = predict_disease(symptoms)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)