from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import threading
import os
import re
import torch
import json
from baseline_model import BaselineModel
from model_LSTM import MedicalModel
from transformers import BertTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TRANSFORMER_PATH = "medical_model"
LSTM_PREFIX      = "current_model"
BASELINE_FILE    = "baseline_model.pkl"
DATASET_FILE     = "medical_dataset.csv"

status = {"current_task": "idle"}


def clean_symptoms(text):
    text = re.sub(
        r"The patient experiences?|Patient (has|reports?|experiences?)",
        "", text, flags=re.IGNORECASE
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def get_disease_details(condition_name):
    try:
        if os.path.exists(DATASET_FILE):
            df = pd.read_csv(DATASET_FILE)
            row = df[df['condition'].str.lower() == condition_name.lower()]
            if not row.empty:
                detail = row.iloc[0]
                return {
                    "warnings":        detail.get('warnings', 'No specific warnings found.'),
                    "recommendations": detail.get('recommendations', 'No specific recommendations found.'),
                    "causes":          detail.get('causes', 'Information not available.')
                }
    except Exception as e:
        print(f"Error fetching details: {e}")
    return {
        "warnings":        "Data not found in dataset.",
        "recommendations": "Data not found in dataset.",
        "causes":          "Data not found in dataset."
    }


@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Medical Diagnosis System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #0f172a; color: white; padding: 20px; font-family: 'Segoe UI', Tahoma, sans-serif; }
        .card { background: #1e293b; border: none; color: white; margin-bottom: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
        .prediction-item { background: #2d3748; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #3182ce; }
        .detail-section { font-size: 0.9rem; margin-top: 10px; padding: 10px; background: #1a202c; border-radius: 5px; }
        .text-warning-custom { color: #f6ad55; font-weight: bold; }
        .text-success-custom { color: #68d391; font-weight: bold; }
        hr { border-color: #4a5568; }
    </style>
</head>
<body>
<div class="container">
    <h2 class="text-center mb-4 text-info">🩺 AI Medical Diagnosis Assistant</h2>
    <div class="card p-4">
        <h5 class="text-primary">🔍 Symptom Analysis & Recommendations</h5><hr>
        <div class="row">
            <div class="col-md-4">
                <select id="predictModel" class="form-select mb-3">
                    <option value="baseline">Baseline (97.98%)</option>
                    <option value="lstm">LSTM (98.79%)</option>
                    <option value="transformer">Transformer (99.99%)</option>
                </select>
            </div>
            <div class="col-md-8">
                <textarea id="symptoms" class="form-control mb-3" rows="2" placeholder="Enter symptoms in English (e.g. fever, dry cough, fatigue)..."></textarea>
            </div>
        </div>
        <button onclick="predict()" class="btn btn-primary w-100">Analyze & Get Results</button>
        <div id="results" class="mt-4"></div>
    </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
    function predict() {
        const resDiv = document.getElementById('results');
        resDiv.innerHTML = '<div class="text-center text-secondary">Analyzing symptoms...</div>';
        fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({
                model_type: document.getElementById('predictModel').value,
                symptoms: document.getElementById('symptoms').value
            })
        }).then(r=>r.json()).then(data => {
            if (data.error) { resDiv.innerHTML=`<div class="alert alert-danger">${data.error}</div>`; return; }
            let h = `<h5>Results using <span class="text-warning">${data.model_used}</span>:</h5>`;
            data.results.forEach(res => {
                h += `
                <div class="prediction-item">
                    <div class="d-flex justify-content-between align-items-center">
                        <strong class="text-info" style="font-size:1.2rem">${res.condition}</strong>
                        <span class="badge bg-primary">${(res.probability*100).toFixed(1)}%</span>
                    </div>
                    <div class="detail-section">
                        <p><span class="text-warning-custom">⚠️ Warnings:</span> ${res.details.warnings}</p>
                        <p><span class="text-success-custom">💡 Recommendations:</span> ${res.details.recommendations}</p>
                        <p><small class="text-secondary">🔬 Possible Causes: ${res.details.causes}</small></p>
                    </div>
                </div>`;
            });
            resDiv.innerHTML = h;
        });
    }
</script>
</body>
</html>
""")


@app.route('/status')
def get_status():
    return jsonify(status)


@app.route('/train', methods=['POST'])
def train_route():
    file   = request.files['file']
    m_type = request.form['model_type']
    path   = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    def worker():
        global status
        status["current_task"] = f"Training {m_type}..."
        try:
            df = pd.read_csv(path)
            df['symptoms'] = df['symptoms'].apply(clean_symptoms)
            if m_type == "baseline":
                m = BaselineModel()
                m.train(df['symptoms'], df['condition'])
                m.save(BASELINE_FILE)
            elif m_type == "lstm":
                m = MedicalModel()
                m.train(df['symptoms'], df['condition'])
                m.save(LSTM_PREFIX)
            status["current_task"] = "Completed ✅"
        except Exception as e:
            status["current_task"] = f"Error: {str(e)}"

    threading.Thread(target=worker).start()
    return jsonify({"message": "Training process started..."})


@app.route('/predict', methods=['POST'])
def predict_route():
    data     = request.json
    m_type   = data['model_type']
    symptoms = clean_symptoms(data['symptoms'])

    try:
        raw_results = []

        if m_type == "baseline":
            m = BaselineModel()
            if m.load(BASELINE_FILE):
                raw_results = m.predict(symptoms)
            else:
                return jsonify({"error": "Baseline model file not found"}), 400

        elif m_type == "lstm":
            m = MedicalModel()
            m.load(LSTM_PREFIX)
            raw_results = m.predict(symptoms)

        elif m_type == "transformer":
            if not os.path.exists(TRANSFORMER_PATH):
                return jsonify({"error": "Transformer model folder missing!"}), 400
            tok = BertTokenizer.from_pretrained(TRANSFORMER_PATH)
            mdl = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_PATH)
            with open(f"{TRANSFORMER_PATH}/labels.json") as f:
                label_map = json.load(f)
            id2label = {str(v): k for k, v in label_map.items()}
            inputs = tok(
                symptoms, return_tensors="pt",
                truncation=True, padding="max_length", max_length=64
            )
            with torch.no_grad():
                probs = torch.softmax(mdl(**inputs).logits, dim=1)[0]
                top3  = torch.topk(probs, 3)
            raw_results = [
                {
                    "condition": id2label[str(int(top3.indices[i]))],
                    "probability": float(top3.values[i])
                }
                for i in range(3)
            ]

        final_results = [
            {
                "condition":   res['condition'],
                "probability": res['probability'],
                "details":     get_disease_details(res['condition'])
            }
            for res in raw_results
        ]
        return jsonify({"model_used": m_type, "results": final_results})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)