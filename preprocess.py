import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize and remove stop words + lemmatization
        words = text.split()
        cleaned_words = [
            self.lemmatizer.lemmatize(w) 
            for w in words 
            if w not in self.stop_words and len(w) > 2
        ]
        
        # Join back
        return " ".join(cleaned_words)

    def prepare_dataset(self, conditions_list):
        df = pd.DataFrame(conditions_list)
        
        # Combine condition name and symptoms for better context
        # Sometimes the name itself contains key keywords
        df['combined_text'] = df['condition'] + " " + df['symptoms']
        df['cleaned_text'] = df['combined_text'].apply(self.clean_text)
        
        return df

    def get_features_labels(self, df):
        # We use cleaned_text as features and condition as labels
        X = df['cleaned_text'].values
        y = df['condition'].values
        return X, y
