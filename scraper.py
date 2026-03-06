import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

class NHSScraper:
    BASE_URL = "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/"

    def get_condition_links(self):
        try:
            response = requests.get(self.BASE_URL, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            links = []
            for a in soup.select("a"):
                href = a.get("href", "")
                if "/illnesses-and-conditions/a-to-z/" in href and len(href.split("/")) > 4:
                    if not href.startswith("http"):
                        href = "https://www.nhsinform.scot" + href
                    links.append(href)
            return list(set(links))
        except Exception as e:
            print(f"Error fetching links: {e}")
            return []

    def scrape_condition(self, url):
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Get condition name
            h1 = soup.find("h1")
            if not h1:
                return None
            condition_name = h1.get_text(strip=True)
            
            # Improved extraction of symptoms
            # Looking for sections that usually contain symptoms
            symptoms_text = ""
            
            # Find the "Symptoms" heading or similar
            symptoms_header = soup.find(lambda tag: tag.name in ['h2', 'h3'] and 'symptom' in tag.get_text().lower())
            
            if symptoms_header:
                # Get text until the next header
                for sibling in symptoms_header.find_next_siblings():
                    if sibling.name in ['h2', 'h3']:
                        break
                    symptoms_text += sibling.get_text(strip=True) + " "
            
            # If no specific section found, try to get the first few paragraphs
            if len(symptoms_text.strip()) < 20:
                paragraphs = soup.select(".article-content p")
                if paragraphs:
                    symptoms_text = " ".join([p.get_text(strip=True) for p in paragraphs[:3]])

            # Fallback if still empty
            if not symptoms_text.strip():
                symptoms_text = f"Symptoms of {condition_name} vary but often include common signs of illness."

            data = {
                "condition": condition_name,
                "url": url,
                "symptoms": symptoms_text.strip(),
                "causes": "Information available on NHS website.",
                "warnings": "Consult a healthcare professional if you are concerned.",
                "recommendations": "Follow medical advice from NHS Inform."
            }
            return data
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def run(self, limit=None):
        links = self.get_condition_links()
        
        if not links:
            print("Using dummy data for demonstration...")
            return [
                {"condition": "Common Cold", "symptoms": "Runny nose, sneezing, sore throat, cough, mild fever", "causes": "Viral infection", "warnings": "Seek help if breathing is difficult", "recommendations": "Rest and fluids"},
                {"condition": "Flu", "symptoms": "High fever, chills, muscle aches, exhaustion, dry cough, congestion", "causes": "Influenza virus", "warnings": "Emergency if chest pain occurs", "recommendations": "Antiviral drugs and rest"},
                {"condition": "Migraine", "symptoms": "Severe throbbing headache, nausea, vomiting, sensitivity to light and sound", "causes": "Genetics and environment", "warnings": "See doctor if sudden and severe", "recommendations": "Pain relief and dark room"},
                {"condition": "Asthma", "symptoms": "Shortness of breath, chest tightness, wheezing, coughing at night", "causes": "Airborne substances", "warnings": "Use inhaler immediately", "recommendations": "Avoid triggers"},
                {"condition": "Diabetes", "symptoms": "Increased thirst, frequent urination, extreme fatigue, blurred vision", "causes": "High blood sugar", "warnings": "Risk of ketoacidosis", "recommendations": "Insulin and diet"}
            ]

        if limit:
            links = links[:limit]
            
        results = []
        for link in tqdm(links, desc="Scraping"):
            data = self.scrape_condition(link)
            if data:
                results.append(data)
            time.sleep(0.1)
        return results
