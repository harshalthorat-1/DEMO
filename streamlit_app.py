import streamlit as st
import pandas as pd
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Heart Stroke Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL STYLING ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# You can create a style.css file for more complex styling
st.markdown("""
<style>
    /* General App Styling */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    /* Main Content Area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Sidebar Styling */
    .st-emotion-cache-16txtl3 {
        background-color: #1e293b;
    }
    /* Title Styling */
    h1 {
        color: #f59e0b;
        text-align: center;
    }
    /* Subheader Styling */
    h2, h3 {
        color: #f59e0b;
    }
    /* Button Styling */
    .stButton>button {
        background-color: #f59e0b;
        color: #0f172a;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #fbbf24;
        color: #0f172a;
    }
    /* Metric Styling */
    .st-emotion-cache-1b0udgb p {
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


# --- LOAD SAVED ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('KNN_heart.pkl')
        scaler = joblib.load('scaler.pkl')
        expected_col = joblib.load('columns.pkl')
        return model, scaler, expected_col
    except FileNotFoundError:
        st.error(
            "Error: Model or helper files not found. Please ensure 'KNN_heart.pkl', 'scaler.pkl', and 'columns.pkl' are in the same directory."
        )
        st.stop()

model, scaler, expected_col = load_artifacts()

# --- HEADER SECTION ---
st.title("❤️ Heart Stroke Prediction Dashboard")
st.markdown("""
Welcome to the Heart Stroke Prediction App. This tool leverages a K-Nearest Neighbors (KNN) model to estimate the risk of heart disease based on your health metrics. 
Please input the patient's details in the sidebar to generate a prediction.
""")

# --- SIDEBAR FOR USER INPUT ---
st.sidebar.header("📋 Patient Details")

def user_input_features():
    """
    Creates sidebar widgets to collect user input for prediction.
    """
    age = st.sidebar.slider("Age", 18, 100, 40, help="Enter the patient's age.")
    sex = st.sidebar.selectbox("Sex", ['M', 'F'], help="Select the patient's gender.")
    chest_pain = st.sidebar.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'], help="Type of chest pain experienced.")
    resting_bp = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120, help="Enter the resting blood pressure.")
    cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200, help="Enter the serum cholesterol level.")
    fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], help="1 for True, 0 for False.")
    resting_ecg = st.sidebar.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'], help="Resting electrocardiogram results.")
    max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150, help="Maximum heart rate achieved during exercise.")
    exercise_angina = st.sidebar.selectbox("Exercise-induced Angina", ["Y", "N"], help="Does the patient experience angina during exercise?")
    oldpeak = st.sidebar.slider('Oldpeak (ST Depression)', 0.0, 6.2, 1.0, 0.1, help="ST depression induced by exercise relative to rest.")
    st_slope = st.sidebar.selectbox('ST Slope', ['Up', 'Flat', 'Down'], help="The slope of the peak exercise ST segment.")

    return {
        'Age': age, 'Sex': sex, 'ChestPainType': chest_pain, 'RestingBP': resting_bp,
        'Cholesterol': cholesterol, 'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
        'MaxHR': max_hr, 'ExerciseAngina': exercise_angina, 'Oldpeak': oldpeak, 'ST_Slope': st_slope
    }

input_dict = user_input_features()

# --- MAIN PANEL LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Your Input Summary")
    # Display the user input in a more structured way
    input_df_display = pd.DataFrame([input_dict])
    st.dataframe(input_df_display.T.rename(columns={0: 'Values'}), use_container_width=True)

# --- DATA PREPROCESSING AND PREDICTION ---
def preprocess_input(input_data, expected_columns):
    """
    Preprocesses the user input to match the model's training data format.
    """
    input_df = pd.DataFrame([input_data])
    input_df_encoded = pd.get_dummies(input_df)
    input_df_aligned = input_df_encoded.reindex(columns=expected_columns, fill_value=0)
    return input_df_aligned

input_df_aligned = preprocess_input(input_dict, expected_col)

with col2:
    st.subheader("📈 Prediction Result")
    if st.sidebar.button('Predict Heart Stroke Risk', key='predict_button', use_container_width=True):
        try:
            # Scale the input data and make a prediction
            input_scaled = scaler.transform(input_df_aligned)
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1]

            # Display the prediction result in a styled card
            if prediction == 1:
                st.error('🔴 High Risk of Heart Disease Detected!', icon="⚠️")
                st.metric(label="Risk Probability", value=f"{proba:.2%}")
            else:
                st.success('🟢 Low Risk of Heart Disease Detected.', icon="✅")
                st.metric(label="Risk Probability", value=f"{proba:.2%}")
            
            st.info("""
            **Disclaimer:** This prediction is based on a machine learning model and is not a substitute for a professional medical diagnosis. 
            Please consult with a qualified healthcare provider for any health concerns.
            """)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.info("Click the 'Predict' button in the sidebar to see the result.")

# --- FOOTER ---
st.markdown("---")
st.markdown("Developed by Harshal Thorat | [LinkedIn](https://www.linkedin.com/in/harshalthorat444)")
