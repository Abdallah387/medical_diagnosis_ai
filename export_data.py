from pymongo import MongoClient
import pandas as pd

# الاتصال بقاعدة البيانات
client = MongoClient("mongodb://localhost:27017/")
db = client["medical_ai"]

# قراءة الكولكشن
data = list(db.conditions.find({}, {"_id": 0}))

# تحويلها إلى dataframe
df = pd.DataFrame(data)

# حفظها CSV
df.to_csv("medical_dataset.csv", index=False)

print("Dataset exported successfully")