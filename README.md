# Medical AI Assistant: NHS Symptoms & LSTM Diagnosis

A deep learning-based medical diagnostic tool that uses **Web Scraping** to gather real-time data from the **NHS Inform** website and an **LSTM Neural Network** to predict conditions based on user symptoms.

## 🚀 Features
- **Real-time Web Scraping:** Automatically extracts medical conditions and symptoms from the NHS website.
- **Deep Learning Model:** Uses a Bidirectional LSTM (Long Short-Term Memory) network for high-accuracy text classification.
- **Interactive Dashboard:** A modern web interface built with Flask and Bootstrap for easy system control and symptom checking.
- **Automated Pipeline:** Integrated Scrape-Train-Predict workflow.

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** TensorFlow, Keras, Scikit-learn
- **Natural Language Processing:** NLTK (Lemmatization, Stop-word removal)
- **Database:** MongoDB (with GridFS for model storage)
- **Web Scraping:** BeautifulSoup4, Requests

## 📂 Project Structure
- `app.py`: Main Flask application with the web dashboard.
- `scraper.py`: NHS web scraping logic.
- `preprocess.py`: Text cleaning and NLP preprocessing.
- `models.py`: LSTM model architecture and training.
- `db_utils.py`: MongoDB interactions and GridFS management.
- `architecture.png`: Visual representation of the system workflow.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/medical-ai.git
   cd medical-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install flask pymongo tensorflow scikit-learn beautifulsoup4 requests tqdm pandas numpy nltk
   ```

3. **Ensure MongoDB is running:**
   The application requires a local MongoDB instance on port 27017.

4. **Run the application:**
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser.

## 📖 How to Use
1. **Scrape:** Click "Scrape NHS Data" to populate the database with real medical data.
2. **Train:** Click "Train AI Model" to build the LSTM model based on the scraped data.
3. **Predict:** Enter symptoms (e.g., "I have a severe headache and nausea") in the text area and click "Get Diagnosis".

## 📊 System Architecture
Refer to `architecture.png` for a detailed view of the data flow and model structure.

---
*Disclaimer: This tool is for educational purposes only and should not be used as a substitute for professional medical advice.*
