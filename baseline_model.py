import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

class BaselineModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
        self.encoder = LabelEncoder()
        self.model = LogisticRegression(max_iter=200)

    def train(self, X, y):
        y_encoded = self.encoder.fit_transform(y)
        X_vec = self.vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y_encoded, test_size=0.1, random_state=42
        )

        self.model.fit(X_train, y_train)
        print("Baseline Accuracy:", accuracy_score(y_test, self.model.predict(X_test)))

    def predict(self, text):
        # تحويل النص باستخدام الـ vectorizer الذي تم تدريبه
        vec = self.vectorizer.transform([text])
        probs = self.model.predict_proba(vec)[0]
        top_idx = probs.argsort()[-3:][::-1]

        results = []
        for idx in top_idx:
            results.append({
                "condition": self.encoder.inverse_transform([idx])[0],
                "probability": float(probs[idx])
            })
        return results

    def save(self, path="baseline_model.pkl"):
        # حفظ القاموس الكامل للمكونات
        with open(path, 'wb') as f:
            pickle.dump({
                "model": self.model,
                "vectorizer": self.vectorizer,
                "encoder": self.encoder
            }, f)

    def load(self, path="baseline_model.pkl"):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.vectorizer = data["vectorizer"]
                self.encoder = data["encoder"]
            return True
        return False