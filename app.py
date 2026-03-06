from flask import Flask, request, jsonify, render_template
from db_utils import MongoDBClient
from scraper import NHSScraper
from preprocess import DataPreprocessor
from models import MedicalModel
import threading
import os
import time
import pickle

app = Flask(__name__)
db = MongoDBClient()
scraper = NHSScraper()
preprocessor = DataPreprocessor()
model_manager = MedicalModel()

status = {"scraping": "idle", "training": "idle"}

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical AI Diagnostics</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .container { max-width: 800px; margin-top: 50px; }
            .card { border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .status-badge { font-size: 0.9rem; padding: 5px 12px; border-radius: 20px; }
            #results { margin-top: 20px; }
            .prediction-card { border-left: 5px solid #0d6efd; margin-bottom: 10px; padding: 15px; background: white; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="text-center mb-5">
                <h1 class="display-4 text-primary">Medical AI Assistant</h1>
                <p class="lead">Diagnosis based on NHS Symptoms & LSTM Model</p>
            </div>

            <div class="card p-4">
                <h5 class="card-title">System Controls</h5>
                <div class="row g-3">
                    <div class="col-md-6">
                        <button onclick="triggerScrape()" class="btn btn-outline-primary w-100">1. Scrape NHS Data</button>
                        <div class="mt-2 small">Scraping: <span id="scrapeStatus" class="badge bg-secondary status-badge">idle</span></div>
                    </div>
                    <div class="col-md-6">
                        <button onclick="triggerTrain()" class="btn btn-outline-success w-100">2. Train AI Model</button>
                        <div class="mt-2 small">Training: <span id="trainStatus" class="badge bg-secondary status-badge">idle</span></div>
                    </div>
                </div>
            </div>

            <div class="card p-4">
                <h5 class="card-title">Symptom Checker</h5>
                <div class="mb-3">
                    <label for="symptoms" class="form-label">Enter your symptoms (in English):</label>
                    <textarea id="symptoms" class="form-control" rows="3" placeholder="e.g., I have a high fever, cough, and feel very tired..."></textarea>
                </div>
                <button onclick="predict()" class="btn btn-primary btn-lg w-100">Get Diagnosis</button>
            </div>

            <div id="results"></div>
        </div>

        <script>
            function updateStatus() {
                fetch('/status').then(r => r.json()).then(data => {
                    const sStatus = document.getElementById('scrapeStatus');
                    const tStatus = document.getElementById('trainStatus');
                    
                    sStatus.innerText = data.scraping;
                    sStatus.className = 'badge status-badge ' + (data.scraping.includes('completed') ? 'bg-success' : data.scraping.includes('running') ? 'bg-warning' : 'bg-secondary');
                    
                    tStatus.innerText = data.training;
                    tStatus.className = 'badge status-badge ' + (data.training.includes('completed') ? 'bg-success' : data.training.includes('running') ? 'bg-warning' : 'bg-secondary');
                });
            }
            setInterval(updateStatus, 2000);

            function triggerScrape() {
                fetch('/scrape', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({limit: 30})
                }).then(r => r.json()).then(d => alert(d.message));
            }

            function triggerTrain() {
                fetch('/train', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                }).then(r => r.json()).then(d => alert(d.message));
            }

            function predict() {
                const symptoms = document.getElementById('symptoms').value;
                if (!symptoms) return alert('Please enter symptoms');
                
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"></div><p>Analyzing...</p></div>';

                fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({symptoms: symptoms})
                }).then(r => r.json()).then(data => {
                    if (data.error) {
                        resultsDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                        return;
                    }
                    
                    let html = '<h3>Predictions:</h3>';
                    data.predictions.forEach(p => {
                        html += `
                            <div class="prediction-card shadow-sm">
                                <div class="d-flex justify-content-between align-items-center">
                                    <h4 class="mb-1 text-primary">${p.condition}</h4>
                                    <span class="badge bg-info text-dark">${(p.probability * 100).toFixed(2)}% Match</span>
                                </div>
                                <p class="mb-2"><strong>Symptoms:</strong> ${p.symptoms || 'N/A'}</p>
                                <p class="mb-1 text-muted small"><strong>Advice:</strong> ${p.recommendations}</p>
                                <a href="${p.url}" target="_blank" class="btn btn-sm btn-link p-0">Read more on NHS website</a>
                            </div>
                        `;
                    });
                    resultsDiv.innerHTML = html;
                });
            }
        </script>
    </body>
    </html>
    """

@app.route('/status')
def get_status():
    return jsonify(status)

@app.route('/scrape', methods=['POST'])
def trigger_scrape():
    limit = request.json.get('limit', 10)
    def run_scrape():
        global status
        status["scraping"] = "running"
        try:
            data = scraper.run(limit=limit)
            for item in data:
                db.save_condition(item)
            status["scraping"] = f"completed: {len(data)} conditions scraped."
        except Exception as e:
            status["scraping"] = f"failed: {str(e)}"
    threading.Thread(target=run_scrape).start()
    return jsonify({"message": "Scraping started"}), 202

@app.route('/train', methods=['POST'])
def trigger_train():
    def run_train():
        global status
        status["training"] = "running"
        try:
            conditions = db.get_all_conditions()
            if len(conditions) < 2:
                status["training"] = "failed: Need more data (at least 2 conditions). Run scrape first."
                return
            
            # 1. Preprocess
            df = preprocessor.prepare_dataset(conditions)
            X, y = preprocessor.get_features_labels(df)
            
            # 2. Train
            model_manager.train(X, y)
            
            # 3. Save locally
            prefix = "current_model"
            model_manager.save(prefix)
            
            # 4. Save to DB
            db.save_model(
                "medical_lstm_v1", 
                "LSTM", 
                prefix, 
                list(model_manager.label_encoder.classes_), 
                {"accuracy": 0.9} # Dummy accuracy for metadata
            )
            
            status["training"] = "completed"
        except Exception as e:
            import traceback
            print(traceback.format_exc()) 
            status["training"] = f"failed: {str(e)}"
    
    threading.Thread(target=run_train).start()
    return jsonify({"message": "Training started"}), 202

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.json.get('symptoms', '')
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400
    
    # Try to load model if not in memory
    if not model_manager.model:
        model_doc = db.load_model("medical_lstm_v1")
        if model_doc:
            # Recreate files from DB
            gridfs_id = model_doc["gridfs_id"]
            with open("current_model_model.h5", "wb") as f:
                f.write(db.fs.get(gridfs_id).read())
            with open("current_model_tokenizer.pkl", "wb") as f:
                f.write(model_doc["tokenizer"])
            with open("current_model_encoder.pkl", "wb") as f:
                f.write(model_doc["encoder"])
            
            model_manager.load("current_model")
        else:
            return jsonify({"error": "Model not found. Please train first."}), 400
            
    # Clean input and predict
    cleaned_input = preprocessor.clean_text(symptoms)
    predictions = model_manager.predict(cleaned_input)
    
    enriched_results = []
    for pred in predictions:
        cond_data = db.conditions.find_one({"condition": pred["condition"]}, {"_id": 0})
        if cond_data:
            pred.update({
                "warnings": cond_data.get("warnings", "No special warnings."),
                "recommendations": cond_data.get("recommendations", ""),
                "causes": cond_data.get("causes", ""),
                "url": cond_data.get("url", "")
            })
        enriched_results.append(pred)
    return jsonify({"predictions": enriched_results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
