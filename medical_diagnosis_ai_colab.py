# -*- coding: utf-8 -*-
"""
medical_diagnosis_ai.py
Run on Google Colab
"""

# Install libraries
# !pip install -U transformers datasets torch scikit-learn pandas accelerate sentencepiece

import pandas as pd
import re
import numpy as np
import torch
import json

from datasets import Dataset
from transformers import (
    BertTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────
# 1. Load dataset
# ──────────────────────────────────────────────
data = pd.read_csv("/content/medical_dataset.csv")
print("Original shape:", data.shape)
print(data['condition'].value_counts())


# ──────────────────────────────────────────────
# 2. Clean symptoms text
# ──────────────────────────────────────────────
def clean_symptoms(text):
    text = re.sub(
        r"The patient experiences?|Patient (has|reports?|experiences?)",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

data['symptoms'] = data['symptoms'].apply(clean_symptoms)


# ──────────────────────────────────────────────
# 3. Balance dataset
# ──────────────────────────────────────────────
target_count = int(data['condition'].value_counts().max())

balanced_dfs = []

for condition in data['condition'].unique():
    df_c = data[data['condition'] == condition]

    df_r = resample(
        df_c,
        replace=True,
        n_samples=target_count,
        random_state=42
    )

    balanced_dfs.append(df_r)

data = pd.concat(balanced_dfs).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nAfter balancing:", data.shape)
print(data['condition'].value_counts())


# ──────────────────────────────────────────────
# 4. Encode labels
# ──────────────────────────────────────────────
unique_conditions = sorted(data['condition'].unique())

label_map = {
    condition: i
    for i, condition in enumerate(unique_conditions)
}

data['labels'] = data['condition'].map(label_map)

num_labels = len(unique_conditions)

print(f"\nNumber of classes: {num_labels}")

with open("labels.json", "w") as f:
    json.dump(label_map, f)


# ──────────────────────────────────────────────
# 5. Tokenizer
# ──────────────────────────────────────────────
model_name = "monologg/biobert_v1.1_pubmed"

tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize(example):
    tokens = tokenizer(
        example["symptoms"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

    tokens["labels"] = example["labels"]
    return tokens


# ──────────────────────────────────────────────
# 6. Dataset
# ──────────────────────────────────────────────
dataset = Dataset.from_pandas(
    data[['symptoms', 'labels']]
)

tokenized_dataset = dataset.map(
    tokenize,
    batched=True
)

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

train_test = tokenized_dataset.train_test_split(
    test_size=0.2,
    seed=42
)

train_dataset = train_test["train"]
test_dataset = train_test["test"]

print(
    f"\nTrain size: {len(train_dataset)}, "
    f"Test size: {len(test_dataset)}"
)


# ──────────────────────────────────────────────
# 7. Model
# ──────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)


# ──────────────────────────────────────────────
# 8. Metrics
# ──────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds)
    }


# ──────────────────────────────────────────────
# 9. Training Arguments
# ──────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./results",

    evaluation_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",

    num_train_epochs=10,
    learning_rate=3e-5,

    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,

    weight_decay=0.01,

    logging_steps=50,

    fp16=torch.cuda.is_available()
)


# ──────────────────────────────────────────────
# 10. Trainer
# ──────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3
        )
    ]
)

trainer.train()


# ──────────────────────────────────────────────
# 11. Save Model
# ──────────────────────────────────────────────
model.save_pretrained("medical_model")
tokenizer.save_pretrained("medical_model")

with open("medical_model/labels.json", "w") as f:
    json.dump(label_map, f)

print("✅ Model saved successfully")