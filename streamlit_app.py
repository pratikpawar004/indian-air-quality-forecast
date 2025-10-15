import os
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from huggingface_hub import hf_hub_download


# PAGE CONFIGURATION
st.set_page_config(page_title="🌫️ AQI Predictor (India Standard)", page_icon="🌫️", layout="centered")

# Global Custom Style
st.markdown("""
    <style>
        .main {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        h1, h2, h3, h4 {
            text-align: center;
        }
        .stButton>button {
            background: linear-gradient(90deg, #3F51B5, #2196F3);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6em 1.5em;
            font-size: 1.1em;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #303F9F, #1976D2);
            transform: scale(1.03);
        }
        div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }
        table {
            margin-left: auto;
            margin-right: auto;
            border-collapse: collapse;
            width: 85%;
            background-color: #2E2E2E;
            color: #FFF;
            border-radius: 10px;
        }
        th, td {
            border: 1px solid #444;
            padding: 10px;
            text-align: center;
        }
        th {
            background-color: #3A3A3A;
        }
    </style>
""", unsafe_allow_html=True)


# ✅ LOAD MODEL (optimized for Streamlit)
@st.cache_resource
def load_model():
    local_path = "best_random_forest_model.joblib"

    # If model already exists locally → load silently
    if os.path.exists(local_path):
        model = joblib.load(local_path)
        return model

    # If not found, download once from Hugging Face
    with st.spinner("📦 Downloading model from Hugging Face Hub..."):
        model_path = hf_hub_download(
            repo_id="pratikpawar004/indian-air-quality-model",
            filename="best_random_forest_model.joblib"
        )
        model = joblib.load(model_path)
        # Save a local copy so it won’t redownload next time
        joblib.dump(model, local_path)
    return model


# ---- LOAD THE MODEL ----
try:
    model = load_model()
    model_features = model.feature_names_in_
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

st.markdown("<h1>🌫️ Air Quality Index Predictor (India)</h1>", unsafe_allow_html=True)
st.write("Predict the Air Quality Index (AQI) using major pollutant concentrations as per Indian CPCB standards.")


# INPUT SECTION

colored_header(label="Input Details", description="Select your city and pollutant concentrations", color_name="blue-70")

cities = [
    'Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal', 'Brajrajnagar',
    'Chandigarh', 'Chennai', 'Coimbatore', 'Delhi', 'Dhule', 'Ernakulam', 'Gurugram',
    'Guwahati', 'Hyderabad', 'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow',
    'Mumbai', 'Pune', 'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram', 'Visakhapatnam', 'Other'
]

city = st.selectbox("🏙️ Select City", cities, index=22)

col1, col2 = st.columns(2)
with col1:
    pm25 = st.number_input("🌁 PM2.5 (µg/m³)", 0.0, 500.0, 49.0, help="Fine particulate matter <2.5μm")
    co = st.number_input("🚗 CO (ppb)", 0.0, 5000.0, 413.0, help="Carbon Monoxide concentration")
    no2 = st.number_input("🌆 NO2 (ppb)", 0.0, 500.0, 15.0, help="Nitrogen Dioxide concentration")
with col2:
    pm10 = st.number_input("🌫️ PM10 (µg/m³)", 0.0, 500.0, 96.0, help="Coarse particulate matter <10μm")
    so2 = st.number_input("🔥 SO2 (ppb)", 0.0, 500.0, 1.0, help="Sulphur Dioxide concentration")
    o3 = st.number_input("☀️ O3 (ppb)", 0.0, 500.0, 21.0, help="Ozone concentration")


# PREDICTION

if st.button("🔮 Predict AQI"):
    input_dict = {feature: 0 for feature in model_features}

    pollutants = {"PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2, "CO": co, "O3": o3}
    for name, value in pollutants.items():
        feature_name = next((f for f in model_features if name.lower() in f.lower()), None)
        if feature_name:
            input_dict[feature_name] = value

    now = datetime.now()
    for f in ["Year", "Month", "IsWeekend"]:
        feature_name = next((feat for feat in model_features if f.lower() in feat.lower()), None)
        if feature_name:
            if f == "IsWeekend":
                input_dict[feature_name] = 1 if now.weekday() >= 5 else 0
            elif f == "Year":
                input_dict[feature_name] = now.year
            else:
                input_dict[feature_name] = now.month

    if city != "Other":
        city_col = f"City_{city}"
        if city_col in input_dict:
            input_dict[city_col] = 1

    input_data = pd.DataFrame([input_dict])[model_features]
    predicted_aqi = float(model.predict(input_data)[0])


    # FIXED CPCB AQI SCALE
        
    if predicted_aqi <= 50:
        category, color, emoji = "Good", "#4CAF50", "🌄💨"        # Clear sky, fresh breeze
    elif predicted_aqi <= 100:
        category, color, emoji = "Satisfactory", "#8BC34A", "🌤️🍃"  # Mild haze, still fresh
    elif predicted_aqi <= 200:
        category, color, emoji = "Moderate", "#FFEB3B", "🌥️"     # Slight haze, neutral tone
    elif predicted_aqi <= 300:
        category, color, emoji = "Poor", "#FF9800", "🌫️😷"         # Visible smog
    elif predicted_aqi <= 400:
        category, color, emoji = "Very Poor", "#F44336", "🔥🚫"      # Air feels heavy, danger rising
    else:
        category, color, emoji = "Severe", "#9C27B0", "☠️🏴"        # Toxic air, extreme danger


    # Display Result
    st.markdown(f"""
        <div style='text-align:center; margin-top:20px;'>
            <h2 style='color:white;'>Predicted AQI</h2>
            <h1 style='color:{color}; font-size:90px; margin:0;'>{predicted_aqi:.0f}</h1>
            <h3 style='color:{color};'>{emoji} {category}</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Pollutant Display
    st.subheader("Pollutant Concentrations")
    col1, col2, col3 = st.columns(3)
    col1.metric("PM2.5 (µg/m³)", f"{pm25:.1f}")
    col2.metric("PM10 (µg/m³)", f"{pm10:.1f}")
    col3.metric("CO (ppb)", f"{co:.1f}")
    col4, col5, col6 = st.columns(3)
    col4.metric("SO₂ (ppb)", f"{so2:.1f}")
    col5.metric("NO₂ (ppb)", f"{no2:.1f}")
    col6.metric("O₃ (ppb)", f"{o3:.1f}")
    style_metric_cards(background_color="#2C2C2C", border_color="#3A3A3A", border_left_color=color)

    st.markdown("---")

    # CPCB AQI SCALE BAR
    st.markdown("<h4 style='text-align:center;'>CPCB AQI Scale (India)</h4>", unsafe_allow_html=True)
    scale_labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    scale_ranges = [50, 100, 200, 300, 400, 500]
    scale_colors = ["#4CAF50", "#8BC34A", "#FFEB3B", "#FF9800", "#F44336", "#9C27B0"]

    scale_html = "<div style='display:flex; height:30px; border-radius:6px; overflow:hidden;'>"
    prev = 0
    for label, max_val, seg_color in zip(scale_labels, scale_ranges, scale_colors):
        width = max_val - prev
        active = prev < predicted_aqi <= max_val
        bg_color = seg_color if active else "#FFFFFF"
        text_color = "white" if active else "#000000"
        scale_html += f"<div style='width:{width/500*100}%; background-color:{bg_color}; text-align:center; line-height:30px; color:{text_color}; font-size:12px;'>{label}</div>"
        prev = max_val
    scale_html += "</div>"
    st.markdown(scale_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CPCB AQI TABLE
    st.markdown("""
    | Category | AQI Range | Health Impact |
    |-----------|------------|---------------|
    | **Good** | 0–50 | Minimal impact |
    | **Satisfactory** | 51–100 | Minor breathing discomfort to sensitive people |
    | **Moderate** | 101–200 | Breathing discomfort to people with lung/heart disease |
    | **Poor** | 201–300 | Breathing discomfort on prolonged exposure |
    | **Very Poor** | 301–400 | Respiratory illness on prolonged exposure |
    | **Severe** | 401–500 | Health emergency, affects healthy people too |
    """)

st.markdown("---")
st.caption("Developed By Pratik Pawar using Streamlit & Random Forest Regressor (CPCB Standard).")
