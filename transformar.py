from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

MODEL_PATH = "medical_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

labels = [
"Flu",
"Common Cold",
"Asthma",
"Migraine",
"Diabetes",
]

def predict_disease(symptoms):
    inputs = tokenizer(
        symptoms,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]

        top3 = torch.topk(probs, 3)

        results = []

        for i in range(3):
            idx = top3.indices[i].item()
            prob = top3.values[i].item()

            results.append({
                "disease": labels[idx],
                "probability": float(prob)
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


@app.route("/demo")
def demo():
    return """ <h1>Medical Diagnosis AI</h1> <input id='symptoms' style='width:400px;height:40px'> <button onclick='predict()'>Predict</button> <pre id='result'></pre>


<script>
async function predict(){
    let symptoms=document.getElementById("symptoms").value

    let response=await fetch("/predict",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({symptoms:symptoms})
    })

    let data=await response.json()

    document.getElementById("result").innerText=
    JSON.stringify(data,null,2)
}
</script>
"""


if __name__ == "__main__":
    app.run(debug=True)
