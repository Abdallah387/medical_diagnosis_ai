import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class DataPreprocessor:

    def __init__(self):
        pass

    def clean_text(self, text):
        # Remove common filler phrases
        text = re.sub(
            r"The patient experiences?|Patient (has|reports?|experiences?)",
            "", text, flags=re.IGNORECASE
        )
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

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

    def balance_dataset(self, df):
        """Oversample minority classes to match the largest class"""
        target_count = df['condition'].value_counts().max()
        balanced_dfs = []
        for condition in df['condition'].unique():
            df_c = df[df['condition'] == condition]
            df_r = resample(df_c, replace=True, n_samples=target_count, random_state=42)
            balanced_dfs.append(df_r)
        return pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    def get_features_labels(self, df):
        return df["symptoms"], df["condition"]

    def split_dataset(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)
