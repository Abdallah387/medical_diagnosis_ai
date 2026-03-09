from baseline_model import BaselineModel
import pandas as pd

data = pd.read_csv("E:\project\medical_diagnosis_ai\medical_dataset.csv")

X = data["symptoms"]
y = data["condition"]

model = BaselineModel()
model.train(X,y)
model.save()