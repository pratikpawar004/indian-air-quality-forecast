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


# CUSTOM PAGE STYLE

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


# LOAD MODEL & SCALER
@st.cache_resource
def load_model():
    local_path = r"D:\data\Projects\AQI\best_model_random_forest.joblib"

    if os.path.exists(local_path):
        data = joblib.load(local_path)
        return data

    with st.spinner("📦 Downloading model from Hugging Face Hub..."):
        model_path = hf_hub_download(
            repo_id="pratikpawar004/indian-air-quality-model",
            filename="best_model_random_forest.joblib"
        )
        data = joblib.load(model_path)
        joblib.dump(data, local_path)  # Save locally for future use
        return data

try:
    model_data = load_model()
    model = model_data["model"]
    scaler = model_data["scaler"]
    model_features = model_data["features"]
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()


# PAGE HEADER

st.markdown("<h1>🌫️ Air Quality Index Predictor (India)</h1>", unsafe_allow_html=True)
st.write("Predict the Air Quality Index (AQI) using pollutant concentrations based on CPCB standards.")

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
    co = st.number_input("🚗 CO (mg/m³)", 0.0, 50.0, 1.0, help="Carbon Monoxide")
    no2 = st.number_input("🌆 NO₂ (µg/m³)", 0.0, 500.0, 40.0, help="Nitrogen Dioxide")

with col2:
    pm10 = st.number_input("🌫️ PM10 (µg/m³)", 0.0, 500.0, 96.0, help="Coarse particulate matter <10μm")
    so2 = st.number_input("🔥 SO₂ (µg/m³)", 0.0, 500.0, 20.0, help="Sulphur Dioxiden")
    o3 = st.number_input("☀️ O₃ (µg/m³)", 0.0, 500.0, 50.0, help="Ozone")

# PREDICTION

if st.button("🔮 Predict AQI"):
    input_dict = {feature: 0 for feature in model_features}

    pollutants = {"PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2, "CO": co, "O3": o3}
    for name, value in pollutants.items():
        feature_name = next((f for f in model_features if name.lower() in f.lower()), None)
        if feature_name:
            input_dict[feature_name] = value

    # Add temporal & city features
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

    input_df = pd.DataFrame([input_dict])[model_features]
    input_scaled = scaler.transform(input_df)
    predicted_aqi = float(model.predict(input_scaled)[0])

   
    # CPCB AQI SCALE
 
    if predicted_aqi <= 50:
        category, color, emoji = "Good", "#4CAF50", "🌄💨"
    elif predicted_aqi <= 100:
        category, color, emoji = "Satisfactory", "#8BC34A", "🌤️🍃"
    elif predicted_aqi <= 200:
        category, color, emoji = "Moderate", "#FFEB3B", "🌥️"
    elif predicted_aqi <= 300:
        category, color, emoji = "Poor", "#FF9800", "🌫️😷"
    elif predicted_aqi <= 400:
        category, color, emoji = "Very Poor", "#F44336", "🔥🚫"
    else:
        category, color, emoji = "Severe", "#9C27B0", "☠️🏴"

  
    # DISPLAY RESULT
   
    st.markdown(f"""
        <div style='text-align:center; margin-top:20px;'>
            <h2 style='color:white;'>Predicted AQI</h2>
            <h1 style='color:{color}; font-size:90px; margin:0;'>{predicted_aqi:.0f}</h1>
            <h3 style='color:{color};'>{emoji} {category}</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 🌫️ Pollutant Summary Section
    st.subheader("Pollutant Concentrations")

    col1, col2, col3 = st.columns(3)
    col1.metric("PM2.5 (µg/m³)", f"{pm25:.1f}")
    col2.metric("PM10 (µg/m³)", f"{pm10:.1f}")
    col3.metric("CO (mg/m³)", f"{co:.1f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("SO₂ (µg/m³)", f"{so2:.1f}")
    col5.metric("NO₂ (µg/m³)", f"{no2:.1f}")
    col6.metric("O₃ (µg/m³)", f"{o3:.1f}")

    style_metric_cards(background_color="#2C2C2C", border_color="#3A3A3A", border_left_color=color)

    st.markdown("---")

    # AQI Scale Bar
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

    # AQI Table
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
