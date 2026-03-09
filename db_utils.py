import os
from pymongo import MongoClient
import gridfs
from datetime import datetime
import pickle

class MongoDBClient:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="medical_ai"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.conditions = self.db["conditions"]
        self.models = self.db["models"]
        self.fs = gridfs.GridFS(self.db)

    def save_condition(self, data):
        return self.conditions.update_one(
            {"condition": data["condition"]},
            {"$set": data},
            upsert=True
        )

    def get_all_conditions(self):
        return list(self.conditions.find({}, {"_id": 0}))

    def save_model(self, name, model_type, path_prefix, labels, metrics):
        # Save the .h5 model file
        model_file = f"{path_prefix}_model.h5"
        if not os.path.exists(model_file):
            print(f"Error: {model_file} not found.")
            return None
            
        with open(model_file, "rb") as f:
            gridfs_id = self.fs.put(f, filename=f"{name}_model.h5")
        
        # Save the tokenizer and encoder to binary
        with open(f"{path_prefix}_tokenizer.pkl", "rb") as f:
            tokenizer_data = f.read()
            
        with open(f"{path_prefix}_encoder.pkl", "rb") as f:
            encoder_data = f.read()
            
        model_doc = {
            "name": name,
            "type": model_type,
            "gridfs_id": gridfs_id,
            "tokenizer": tokenizer_data,
            "encoder": encoder_data,
            "labels": labels,
            "metrics": metrics,
            "created": datetime.now()
        }
        return self.models.update_one({"name": name}, {"$set": model_doc}, upsert=True)

    def load_model(self, name):
        return self.models.find_one({"name": name})
