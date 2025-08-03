import streamlit as st
import pandas as pd
import joblib

# --- LOAD SAVED ARTIFACTS ---
try:
    model = joblib.load('KNN_heart.pkl')
    scaler = joblib.load('scaler.pkl')
    expected_col = joblib.load('columns.pkl')
except FileNotFoundError:
    st.error(
        "Error: Model or helper files not found. Please ensure 'KNN_heart.pkl', 'scaler.pkl', and 'columns.pkl' are in the same directory."
    )
    st.stop()

# --- STREAMLIT APP USER INTERFACE ---
st.set_page_config(page_title="Heart Stroke Prediction", page_icon="❤️", layout="wide")

st.title("❤️ Heart Stroke Prediction App")
st.markdown("""
This application uses a K-Nearest Neighbors (KNN) model to predict the risk of a heart stroke based on user-provided health metrics.
Please fill in the details on the left sidebar to get a prediction.
""")

st.sidebar.header("Enter Patient Details")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 100, 40)
    sex = st.sidebar.selectbox("Sex", ['M', 'F'])
    chest_pain = st.sidebar.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
    resting_bp = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], help="0 = False, 1 = True")
    resting_ecg = st.sidebar.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
    max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.sidebar.selectbox("Exercise-induced Angina", ["Y", "N"])
    oldpeak = st.sidebar.slider('Oldpeak (ST Depression)', 0.0, 6.2, 1.0, 0.1)
    st_slope = st.sidebar.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

    return {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

input_dict = user_input_features()

st.subheader("Your Input Parameters")
st.write(pd.DataFrame([input_dict]))

# --- DATA PREPROCESSING AND PREDICTION ---
input_df = pd.DataFrame([input_dict])
input_df_encoded = pd.get_dummies(input_df)
input_df_aligned = input_df_encoded.reindex(columns=expected_col, fill_value=0)

# --- PREDICTION LOGIC ---
if st.button('Predict Heart Stroke Risk', key='predict_button'):
    try:
        input_scaled = scaler.transform(input_df_aligned)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error('🔴 High Risk of Heart Disease Detected!', icon="⚠️")
            st.metric(label="Risk Probability", value=f"{proba:.2%}")
            st.warning("Disclaimer: This is a prediction based on a machine learning model and not a medical diagnosis. Please consult a healthcare professional.")
        else:
            st.success('🟢 Low Risk of Heart Disease Detected.', icon="✅")
            st.metric(label="Risk Probability", value=f"{proba:.2%}")
            st.info("Disclaimer: This prediction is for informational purposes only. Regular check-ups with a doctor are always recommended.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
