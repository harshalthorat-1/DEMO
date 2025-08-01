# Heart Stroke Prediction App

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5.1-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A user-friendly web application built with Streamlit to predict the risk of heart disease based on various health parameters. The prediction is powered by a K-Nearest Neighbors (KNN) machine learning model trained on the Heart Disease dataset.

---

## 🚀 Live Demo

You can access the live application deployed on Streamlit Cloud here:

**[➡️ Heart Stroke Prediction App]([https://your-streamlit-app-url.streamlit.app/](https://blank-app-bvy4otu2h0h.streamlit.app/))** 

---

## ✨ Features

- **Interactive UI**: A simple and intuitive interface for users to input their health data.
- **Instant Predictions**: Get real-time predictions on the likelihood of heart disease.
- **High Accuracy**: The underlying KNN model achieves an **accuracy of 88.59%** on the test set.
- **Data-Driven**: Trained on a comprehensive dataset featuring 11 key health attributes.
- **Accessible**: Easily deployable and accessible via any web browser.

---

## 📊 Dataset

This project utilizes the **Heart Disease Dataset**, which contains 918 records and 12 attributes related to heart health. The model was trained on the following features:

- **Age**: Age of the patient [years]
- **Sex**: Sex of the patient [M: Male, F: Female]
- **ChestPainType**: Chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- **RestingBP**: Resting blood pressure [mm Hg]
- **Cholesterol**: Serum cholesterol [mm/dl]
- **FastingBS**: Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- **RestingECG**: Resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality, LVH: showing probable or definite left ventricular hypertrophy]
- **MaxHR**: Maximum heart rate achieved [Numeric value between 60 and 202]
- **ExerciseAngina**: Exercise-induced angina [Y: Yes, N: No]
- **Oldpeak**: Oldpeak = ST [Numeric value measured in depression]
- **ST_Slope**: The slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- **HeartDisease**: The target variable indicating the presence of heart disease [1: Heart Disease, 0: Normal]

---

## 🤖 Machine Learning Model

The prediction model was developed through a standard data science pipeline involving data cleaning, preprocessing, model selection, and evaluation.

### 1. Data Preprocessing

- **Handling Missing Values**: Rows with missing or zero values for critical attributes like `Cholesterol` and `RestingBP` were imputed with the mean of the respective columns.
- **Encoding Categorical Data**: Categorical features (e.g., `Sex`, `ChestPainType`) were converted into a numerical format using one-hot encoding.
- **Feature Scaling**: All numerical features were scaled using `StandardScaler` to ensure that no single feature disproportionately influences the model's predictions.

### 2. Model Selection & Training

Several classification models were trained and evaluated on the preprocessed data:
- Logistic Regression
- **K-Nearest Neighbors (KNN)**
- Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

The **K-Nearest Neighbors (KNN)** model was selected as the final model due to its superior performance.

### 3. Evaluation

The chosen KNN model achieved the following results on the unseen test data:
- **Accuracy**: **88.59%**
- **F1 Score**: **89.86%**

The trained model and the scaler object have been saved to `model.pkl` for use in the Streamlit application.

---

## 🛠️ Technology Stack

- **Language**: Python 3.9
- **Libraries**:
  - Scikit-learn (for model building and evaluation)
  - Pandas (for data manipulation)
  - Streamlit (for creating the web interface)
  - Joblib (for saving the model)
  - Sheryanalysis (for initial EDA)

---

## ⚙️ How to Run Locally

To run this application on your local machine, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/heart-stroke-prediction-app.git](https://github.com/your-username/heart-stroke-prediction-app.git)
    cd heart-stroke-prediction-app
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    Make sure you have a `requirements.txt` file with the necessary libraries.
    ```
    # requirements.txt
    streamlit
    pandas
    scikit-learn
    joblib
    ```
    Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser.

---

## 🚀 Deployment

This application is deployed using **Streamlit Cloud**. Deployment is straightforward:
1.  Push the project code (including `app.py`, `model.pkl`, and `requirements.txt`) to a GitHub repository.
2.  Connect your GitHub account to Streamlit Cloud.
3.  Select the repository and deploy the app.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author


**Harshal Thorat**  
- **LinkedIn**: [Harshal](www.linkedin.com/in/harshalthorat444)

Feel free to reach out with any questions or feedback!
