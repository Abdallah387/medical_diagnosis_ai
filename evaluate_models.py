import pandas as pd
import numpy as np
import re
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers

from baseline_model import BaselineModel
from model_LSTM import MedicalModel


def clean_symptoms(text):
    text = re.sub(
        r"The patient experiences?|Patient (has|reports?|experiences?)",
        "", text, flags=re.IGNORECASE
    )
    return re.sub(r"\s+", " ", text).strip().lower()


# ── Load & clean dataset ──────────────────────────────────────
data = pd.read_csv("medical_dataset.csv")
data['symptoms'] = data['symptoms'].apply(clean_symptoms)

X = data["symptoms"]
y = data["condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []


# ── Baseline ─────────────────────────────────────────────────
print("Evaluating Baseline...")
baseline = BaselineModel()
baseline.train(X_train, y_train)

preds = [baseline.predict(t)[0]["condition"] for t in X_test]
results.append({
    "model": "TFIDF + Logistic Regression",
    "accuracy": accuracy_score(y_test, preds),
    "f1": f1_score(y_test, preds, average="macro")
})


# ── LSTM ─────────────────────────────────────────────────────
print("Evaluating LSTM...")
lstm = MedicalModel()
lstm.train(X_train, y_train)

preds = [lstm.predict(t)[0]["condition"] for t in X_test]
results.append({
    "model": "LSTM",
    "accuracy": accuracy_score(y_test, preds),
    "f1": f1_score(y_test, preds, average="macro")
})


# ── Transformer ───────────────────────────────────────────────
print("Evaluating Transformer...")
MODEL_PATH = r"E:\project\medical_diagnosis_ai - Copy (2)\medical_model"

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

with open(f"{MODEL_PATH}/labels.json") as f:
    label_map = json.load(f)
id2label = {v: k for k, v in label_map.items()}

preds = []
for text in X_test:
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding="max_length", max_length=64
    )
    with torch.no_grad():
        pred = torch.argmax(model(**inputs).logits, dim=1).item()
    preds.append(id2label.get(pred, "Unknown"))

results.append({
    "model": "BioBERT Transformer",
    "accuracy": accuracy_score(y_test, preds),
    "f1": f1_score(y_test, preds, average="macro")
})


# ── Print results ─────────────────────────────────────────────
print("\n{'='*50}")
print("Model Comparison")
print('='*50)
for r in results:
    print(f"{r['model']:<35} Accuracy: {r['accuracy']:.3f}   F1: {r['f1']:.3f}")
