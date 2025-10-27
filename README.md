# 🌫️ Air Quality Index (AQI) Predictor — India Standard 🇮🇳

![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit)
![Model](https://img.shields.io/badge/Model-RandomForestRegressor-2C7BB6)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

## 📘 Overview

**Air Quality Index Predictor** is a machine learning web app built with **Streamlit** that predicts the **Air Quality Index (AQI)** based on pollutant concentrations like PM2.5, PM10, NO₂, SO₂, CO, and O₃ — following **Central Pollution Control Board: CPCB
 (India)** standards.  
It provides real-time predictions, health impact analysis, and a beautifully visualized AQI scale.

---

## 🚀 Live Demo

🔗 **[Try it on Streamlit Cloud](https://aqi-predictor-india.streamlit.app/)** 

Adjust the input parameters — **PM2.5 (µg/m³), PM10 (µg/m³), NO₂ (ppb), SO₂ (ppb), CO (ppb), and O₃ (ppb)** — based on reference values from [AQI India Dashboard (Pune)](https://www.aqi.in/in/dashboard/india/maharashtra/pune) to predict the Air Quality Index using a trained Random Forest model.


---
# 🧩 Model Details

- Algorithm: Random Forest Regressor

- Trained On: Indian Air Quality Dataset

- Model File: best_random_forest_model.joblib

- Hosted On: [Hugging Face Hub](https://huggingface.co/pratikpawar004/indian-air-quality-model)


- The model is automatically downloaded from Hugging Face when not found locally.

---
## ⚙️ Features

✅ Predicts AQI based on pollutant concentration levels  
✅ Displays AQI category and health impact  
✅ City-based inputs with **pollutant value sliders (numeric input)**  
✅ Visualization of pollutant contribution  
✅ Model hosted on **Hugging Face Hub** and loaded dynamically  
✅ Lightweight, fast, and deployable on Streamlit Cloud  

---

## 🏗️ Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| Backend | Python |
| ML Model | RandomForestRegressor |
| Deployment | Streamlit Cloud / Localhost |
| Storage | Hugging Face Hub (Model Hosting) |

---

## 📂 Project Structure
```
📦 Air Quality Index Predictor
│
├── streamlit_app.py # Main Streamlit app
├── city_day.csv # Dataset file
├── requirements.txt # Python dependencies
├── AQI # Jupyter Notebook File
├── best_random_forest_model.joblib # Trained ML model (excluded in repo)
└── README.md # Project documentation
```
## ⚙️ Installation Guide

Follow these steps to set up the project locally 👇

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/pratikpawar004/indian-air-quality-forecast.git
cd indian-air-quality-forecast
```
### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```
python -m venv venv
venv\Scripts\activate      # For Windows
# OR
source venv/bin/activate   # For macOS/Linux
```
### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 4️⃣ Run the Streamlit App
```
streamlit run streamlit_app.py
```

# 👨‍💻 Author
## Pratik Pawar  
