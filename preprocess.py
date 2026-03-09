import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split


class DataPreprocessor:

    def __init__(self):
        pass


    # تنظيف النص
    def clean_text(self, text):

        text = text.lower()

        text = re.sub(r'\d+', '', text)

        text = text.translate(
            str.maketrans('', '', string.punctuation)
        )

        text = re.sub(r'\s+', ' ', text).strip()

        return text


    # تجهيز الداتا من MongoDB
    def prepare_dataset(self, conditions):

        records = []

        for cond in conditions:

            condition_name = cond["condition"]

            symptoms = cond.get("symptoms", [])

            if isinstance(symptoms, str):
                symptoms = [symptoms]

            for s in symptoms:

                cleaned = self.clean_text(s)

                if len(cleaned) > 3:

                    records.append({
                        "symptoms": cleaned,
                        "condition": condition_name
                    })

        df = pd.DataFrame(records)

        return df


    # استخراج features و labels
    def get_features_labels(self, df):

        X = df["symptoms"]
        y = df["condition"]

        return X, y


    # تقسيم الداتا
    def split_dataset(self, X, y):

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )