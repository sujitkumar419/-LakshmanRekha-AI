import streamlit as st
import numpy as np
import joblib

# Page configuration framework initialization
st.set_page_config(
    page_title="LakshmanRekha AI",
    page_icon="🛡️",
    layout="centered"
)


# Load production serialization artifacts
@st.cache_resource
def load_production_artifacts():
    model = joblib.load('lakshmanrekha_svr_model.pkl')
    scaler_x = joblib.load('scaler_x.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_x, scaler_y


try:
    model, scaler_x, scaler_y = load_production_artifacts()
except FileNotFoundError:
    st.error("Deployment Artifacts Missing! Please check the working directory.")
    st.stop()

# --- DUAL LANGUAGE DICTIONARY LOOKUP ---
# Custom dictionary mapping for seamless UI language toggle
lang_dict = {
    "English": {
        "title": "🛡️ LakshmanRekha AI",
        "subtitle": "SVR-Powered Human Tolerance & Frustration Tracker",
        "desc": "This predictive framework utilizes Support Vector Regression to model behavioral thresholds. It implements an $\epsilon$-insensitive tube boundary to absorb baseline stress deviations while accurately mapping extreme variance outliers.",
        "lbl_hours": "Daily Working Hours",
        "lbl_salary": "Salary Satisfaction Level (1-Low, 5-High)",
        "lbl_manager": "Manager Relationship Index (1-Best, 5-Worst)",
        "lbl_exp": "Experience Tenure (Years)",
        "lbl_promo": "Years Since Last Promotion",
        "btn_evaluate": "Evaluate Behavior Metric Pipeline",
        "lbl_metric": "Calculated Cumulative Frustration Index",
        "msg_green": "🟢 **Stable State:** Data vector resides safely inside the $\epsilon$-tube boundaries. Natural variance is fully absorbed.",
        "msg_yellow": "🟡 **Warning State:** Approaching critical tolerance thresholds. System friction is escalating. Intervention recommended.",
        "msg_red": "🔴 **Breach Alert:** Boundary violated! Subject functions as a critical Support Vector outlier. High attrition probability."
    },
    "Hindi / Hinglish": {
        "title": "🛡️ लक्ष्मणरेखा AI (LakshmanRekha AI)",
        "subtitle": "SVR-पावर्ड ह्यूमन टॉलरेंस और फ्रस्ट्रेशन ट्रैकर",
        "desc": "यह AI मॉड्यूल Support Vector Regression (SVR) का उपयोग करके इंसान के सब्र की सीमा को मापता है। यह एक 'सुरक्षा घेरा' ($\epsilon$-tube) बनाता है जो रोज़मर्रा के छोटे-मोटे तनाव को माफ कर देता है, लेकिन लिमिट टूटने पर आउटलियर्स (Support Vectors) को तुरंत पकड़ लेता है।",
        "lbl_hours": "रोज़ाना काम के घंटे (Daily Working Hours)",
        "lbl_salary": "सैलरी से संतुष्टि (1-कम, 5-सबसे ज़्यादा)",
        "lbl_manager": "मैनेजर के साथ रिश्ता (1-सबसे अच्छा, 5-खराब)",
        "lbl_exp": "कंपनी में अनुभव (कुल वर्ष)",
        "lbl_promo": "पिछले प्रमोशन से अब तक का समय (वर्ष)",
        "btn_evaluate": "बिहेवियर पल्स का विश्लेषण करें 🔥",
        "lbl_metric": "कैलकुलेटेड कुल फ्रस्ट्रेशन इंडेक्स (Frustration Index)",
        "msg_green": "🟢 **सुरक्षित स्थिति (Safe Zone):** इंसान बिल्कुल सुरक्षा घेरे के अंदर है। छोटी-मोटे उतार-चढ़ाव माफ हो रहे हैं। सब कुछ सामान्य है।",
        "msg_yellow": "🟡 **चेतावनी की स्थिति (Warning Zone):** सब्र की सीमा समाप्त होने वाली है। सिस्टम में फ्रस्ट्रेशन बढ़ रहा है। तुरंत ध्यान देने की ज़रूरत है।",
        "msg_red": "🔴 **लक्ष्मणरेखा पार (Breach Alert):** लिमिट पूरी तरह टूट चुकी है! इंसान सिस्टम के बाहर (Outlier) निकल गया है। कंपनी या साथ छोड़ने का भारी खतरा है।"
    }
}

# Language Configuration Toggle Selection
selected_lang = st.selectbox("🌐 Choose UI Language / भाषा चुनें", ["English", "Hindi / Hinglish"])
text = lang_dict[selected_lang]

# Render Dynamic UI elements based on selection
st.title(text["title"])
st.subheader(text["subtitle"])
st.markdown(text["desc"])
st.markdown("---")

# Feature Input Canvas Layout
col1, col2 = st.columns(2)

with col1:
    daily_hours = st.slider(text["lbl_hours"], 6.0, 12.0, 8.5, step=0.1)
    salary_satisfaction = st.slider(text["lbl_salary"], 1, 5, 3)
    manager_relation = st.slider(text["lbl_manager"], 1, 5, 3)

with col2:
    experience_years = st.number_input(text["lbl_exp"], min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    promotion_gap = st.number_input(text["lbl_promo"], min_value=0.0, max_value=5.0, value=1.0, step=0.5)

st.markdown("---")

# Pipeline Processing Execution Trigger
if st.button(text["btn_evaluate"], use_container_width=True):
    # Map layout input fields to numerical raw matrices
    raw_input = np.array([[daily_hours, experience_years, salary_satisfaction, promotion_gap, manager_relation]])

    # Feature transform operations
    scaled_input = scaler_x.transform(raw_input)
    scaled_prediction = model.predict(scaled_input)
    final_score = float(scaler_y.inverse_transform(scaled_prediction.reshape(-1, 1)).ravel())

    # Render interactive evaluation output block
    st.metric(label=text["lbl_metric"], value=f"{final_score:.2f} %")

    # Core classification alert dispatch logic
    if final_score <= 45.0:
        st.success(text["msg_green"])
    elif 45.0 < final_score <= 75.0:
        st.warning(text["msg_yellow"])
    else:
        st.error(text["msg_red"])
