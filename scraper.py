import requests
from bs4 import BeautifulSoup
import random
import json
from tqdm import tqdm

BASE = "https://www.nhsinform.scot"

headers = {
    "User-Agent": "Mozilla/5.0"
}

# ---------------------------
# Step 1: get condition links
# ---------------------------
def get_conditions(limit=18):

    url = BASE + "/illnesses-and-conditions/a-to-z/"
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    links = []
    for a in soup.select("main a"):

        href = a.get("href")
        name = a.text.strip()

        if href and "/illnesses-and-conditions/" in href and len(name) > 3:
            links.append({
                "name": name,
                "url": BASE + href
            })

        if len(links) >= limit:
            break

    return links


# ---------------------------
# Step 2: extract symptoms
# ---------------------------
def get_symptoms(url):

    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    symptoms = []

    for li in soup.select("li"):
        text = li.text.strip()

        if len(text) > 5 and len(text) < 120:
            symptoms.append(text)

    return list(set(symptoms))[:10]


# ---------------------------
# Step 3: generate patient reports
# ---------------------------
def generate_reports(conditions, total_cases=8000):

    reports = []

    per_disease = total_cases // len(conditions)

    for cond in conditions:

        symptoms = cond["symptoms"]

        for _ in range(per_disease):

            sample = random.sample(symptoms, min(3, len(symptoms)))

            report = "I have " + ", ".join(sample)

            reports.append({
                "text": report,
                "disease": cond["name"]
            })

    return reports


# ---------------------------
# MAIN
# ---------------------------

print("Collecting diseases...")
conditions = get_conditions(18)

print("Extracting symptoms...")
for cond in tqdm(conditions):

    cond["symptoms"] = get_symptoms(cond["url"])


print("Generating 8000 patient reports...")
data = generate_reports(conditions, 8000)

print("Saving dataset...")

with open("medical_dataset_8000.json", "w") as f:
    json.dump(data, f, indent=2)

print("DONE!")
print("Total samples:", len(data))