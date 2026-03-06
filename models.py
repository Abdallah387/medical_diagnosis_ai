import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class MedicalModel:
    def __init__(self, max_words=5000, max_len=150):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.label_encoder = LabelEncoder()
        self.model = None

    def build_lstm(self, num_classes):
        model = Sequential([
            Embedding(self.max_words, 128, input_length=self.max_len),
            SpatialDropout1D(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        # Use a smaller learning rate for stability
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def train(self, X, y):
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)
        
        # Tokenize text
        self.tokenizer.fit_on_texts(X)
        sequences = self.tokenizer.texts_to_sequences(X)
        X_padded = pad_sequences(sequences, maxlen=self.max_len)
        
        # Data Augmentation: Duplicate small datasets to allow training
        if len(X_padded) < 20:
            print("Data too small, augmenting by duplication...")
            X_padded = np.tile(X_padded, (10, 1))
            y_encoded = np.tile(y_encoded, (10,))
            
        # Split data
        test_size = 0.2 if len(X_padded) > 10 else 0.1
        X_train, X_val, y_train, y_val = train_test_split(X_padded, y_encoded, test_size=test_size, random_state=42)
        
        # Build and train
        self.build_lstm(num_classes)
        
        # Use early stopping to prevent overfitting on small data
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        epochs = 30 if len(X_padded) > 100 else 50
        self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=min(32, len(X_train)), 
            validation_data=(X_val, y_val), 
            verbose=1,
            callbacks=[early_stop]
        )
        
        return self.model

    def predict(self, text):
        if not self.model:
            return []
            
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len)
        pred = self.model.predict(padded, verbose=0)
        
        # Get top 3 predictions
        top_idx = np.argsort(pred[0])[-3:][::-1]
        results = []
        for idx in top_idx:
            condition_name = self.label_encoder.inverse_transform([idx])[0]
            try:
                condition_name = str(condition_name)
            except:
                pass
                
            results.append({
                "condition": condition_name,
                "probability": float(pred[0][idx])
            })
        return results

    def save(self, path_prefix):
        self.model.save(f"{path_prefix}_model.h5")
        with open(f"{path_prefix}_tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)
        with open(f"{path_prefix}_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

    def load(self, path_prefix):
        self.model = tf.keras.models.load_model(f"{path_prefix}_model.h5")
        with open(f"{path_prefix}_tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(f"{path_prefix}_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
