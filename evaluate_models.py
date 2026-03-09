import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from baseline_model import BaselineModel
from model_LSTM import MedicalModel

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# load dataset
data = pd.read_csv("E:\project\medical_diagnosis_ai\medical_dataset.csv")

X = data["symptoms"]
y = data["condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


results = []

# --------------------
# Baseline Model
# --------------------

baseline = BaselineModel()

baseline.train(X_train, y_train)

preds = []

for text in X_test:
    p = baseline.predict(text)[0]["condition"]
    preds.append(p)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")

results.append({
    "model": "TFIDF + Logistic Regression",
    "accuracy": acc,
    "f1": f1
})


# --------------------
# LSTM Model
# --------------------

lstm = MedicalModel()

lstm.train(X_train, y_train)

preds = []

for text in X_test:

    p = lstm.predict(text)[0]["condition"]
    preds.append(p)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")

results.append({
    "model": "LSTM",
    "accuracy": acc,
    "f1": f1
})


# --------------------
# Transformer Model
# --------------------

MODEL_PATH = "E:\project\medical_diagnosis_ai\medical_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cpu")

model.to(device)
model.eval()

import json

# قراءة labels
with open("medical_model/labels.json") as f:
    label_map = json.load(f)

# تحويل المفاتيح لـ int عشان التوافق مع pred
# label_map غالبًا شكلها { "0": "Hypertension", ... }
id2label = {int(k): v for k, v in label_map.items()}

preds = []

for text in X_test:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

        # حماية من أي KeyError
        if pred in id2label:
            preds.append(id2label[pred])
        else:
            preds.append("Unknown")
            print(f"Warning: prediction {pred} not found in id2label!")

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")

results.append({
    "model": "BioBERT Transformer",
    "accuracy": acc,
    "f1": f1
})


# --------------------
# Print Results
# --------------------

print("\nModel Comparison\n")

for r in results:

    print(
        r["model"],
        "Accuracy:", round(r["accuracy"],3),
        "F1:", round(r["f1"],3)
    )