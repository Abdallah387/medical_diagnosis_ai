import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

st.title("Medical Diagnosis AI")

st.write("Enter your symptoms and the AI model will predict possible diseases.")

symptoms = st.text_input(
"Enter symptoms",
"Example: fever cough headache"
)

def predict(symptoms):
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

            results.append((labels[idx], prob))

        return results


if st.button("Predict Disease"):

    results = predict(symptoms)
 
    st.subheader("Top Predictions")

    for disease, prob in results:

        st.write(f"{disease} : {prob:.2f}")

