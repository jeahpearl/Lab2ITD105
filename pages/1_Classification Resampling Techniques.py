import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Preload Lung Cancer Dataset
lung_cancer_path = r"C:\Users\jeahp\Desktop\ITD105\Lab2\csv\survey lung cancer.csv"
lung_cancer_data = pd.read_csv(lung_cancer_path)

# Specify the folder path where models should be saved
FOLDER_PATH = "models"

# Set the title of the app
st.title("Classification Sampling Technique")

# Encode categorical features into numeric and map binary features correctly
def preprocess_data(dataframe):
    binary_features = [
        "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC_DISEASE",
        "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING",
        "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN"
    ]
    
    for column in binary_features:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].map({2: 1, 1: 0})
    
    label_encoders = {}
    for column in dataframe.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        dataframe[column] = le.fit_transform(dataframe[column])
        label_encoders[column] = le

    return dataframe, label_encoders

# Save the model locally
def save_model_locally(model, filename):
    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH)
    file_path = os.path.join(FOLDER_PATH, filename)
    with open(file_path, "wb") as f:
        joblib.dump(model, f)
    return file_path

# K-Fold Cross-Validation
def kfold_cv():
    st.header("1. Train Model with K-Fold Cross Validation")
    st.write("Using Preloaded Lung Cancer Dataset")
    
    # Display preview of dataset
    st.subheader("Dataset Preview")
    with st.expander("Data Preview"):
        st.dataframe(lung_cancer_data)

    # Preprocess the data
    dataframe, label_encoders = preprocess_data(lung_cancer_data)
    st.subheader("Dataset After Encoding")
    with st.expander("Data Preview"):
        st.dataframe(dataframe)

    X = dataframe.iloc[:, :-1].values  # Features (all columns except the last one)
    Y = dataframe.iloc[:, -1].values  # Target (last column)

    num_folds = st.slider("Select number of folds for K-Fold Cross Validation:", 2, 10, 5)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=210)
    results = cross_val_score(model, X, Y, cv=kfold)

    st.subheader("K-Fold Cross-Validation Results")
    st.write(f"Accuracy: {results.mean() * 100:.3f}%")
    st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

    model.fit(X, Y)
    st.success("The model has been trained and is ready for download.")

    if st.button("Download the Trained Model"):
        model_folder = r"C:\Users\user\Desktop\jeah\ITD105\LABORATORY2\Models"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        model_filename = os.path.join(model_folder, "KFOLD_lung_cancer_survey_model.joblib")
        joblib.dump(model, model_filename)
        st.success(f"Model trained and saved as {model_filename}")

# Leave-One-Out Cross-Validation
def loocv():
    st.header("2. Train Model with Leave-One-Out Cross Validation (LOOCV)")
    st.write("Using Preloaded Lung Cancer Dataset")

    # Display preview of dataset
    st.subheader("Dataset Preview")
    with st.expander("Data Preview"):
        st.dataframe(lung_cancer_data)

    # Preprocess the data
    dataframe, label_encoders = preprocess_data(lung_cancer_data)
    X = dataframe.iloc[:, :-1].values  # Features (all columns except the last one)
    Y = dataframe.iloc[:, -1].values  # Target (last column)

    st.subheader("Dataset After Encoding")
    with st.expander("Data Preview"):
        st.dataframe(dataframe)

    st.write("Training Leave One Out Cross Validation Model...")
    loocv = LeaveOneOut()
    model = LogisticRegression(max_iter=500)
    results = cross_val_score(model, X, Y, cv=loocv)

    st.subheader("Leave-One-Out Cross-Validation Results")
    st.write(f"Accuracy: {results.mean() * 100:.3f}%")
    st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

    model.fit(X, Y)
    st.success("The model has been trained and is ready for download.")

    if st.button("Download the Trained Model"):
        model_folder = r"C:\Users\jeahp\Desktop\ITD105\Lab2\Models"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        model_filename = os.path.join(model_folder, "LOOCV_lung_cancer_survey_model.joblib")
        joblib.dump(model, model_filename)
        st.success(f"Model trained and saved as {model_filename}")

# Prediction
def predictor():
    st.header("Lung Cancer Diagnosis Predictor")

    # Define paths for pre-trained models
    kfold_model_path = r"C:\Users\jeahp\Desktop\ITD105\Lab2\Models\KFOLD_lung_cancer_survey_model.joblib"
    loocv_model_path = r"C:\Users\jeahp\Desktop\ITD105\Lab2\Models\LOOCV_lung_cancer_survey_model.joblib"

    # Dropdown for model selection
    model_choice = st.selectbox(
        "Choose a pre-trained model:",
        options=["K-Fold Cross-Validation Model", "Leave-One-Out Cross Validation Model"]
    )

    # Load the selected model with error handling
    model = None
    try:
        if model_choice == "K-Fold Cross-Validation Model":
            model = joblib.load(kfold_model_path)
            st.success("K-Fold Cross-Validation Model loaded successfully!")
        elif model_choice == "Leave-One-Out Cross Validation Model":
            model = joblib.load(loocv_model_path)
            st.success("Leave-One-Out Cross Validation Model loaded successfully!")
    except FileNotFoundError:
        st.error(f"Error: The selected model file could not be found. Please ensure the model is located at:\n{model_choice}")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        return

    st.subheader("Input Sample Data for Prediction")

    # Row 1: Gender and Age
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        gender = 1 if gender == "M" else 0  # Map Male to 1, Female to 0
    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, value=1)

    # Row 2: Smoking and Yellow Fingers
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        smoking = 1 if smoking == "Yes" else 0  # Match remapping logic
    with col2:
        yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
        yellow_fingers = 1 if yellow_fingers == "Yes" else 0
    with col3:
        anxiety = st.selectbox("Anxiety", ["Yes", "No"])
        anxiety = 1 if anxiety == "Yes" else 0
    with col4:
        alcohol = st.selectbox("Alcohol Consumption", ["Yes", "No"])
        alcohol = 1 if alcohol == "Yes" else 0

    # Row 3: Chronic Disease, Chest Pain, Peer Pressure
    col1, col2, col3 = st.columns(3)
    with col1:
        chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
        chronic_disease = 1 if chronic_disease == "Yes" else 0
    with col2:
        chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])
        chest_pain = 1 if chest_pain == "Yes" else 0
    with col3:
        peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
        peer_pressure = 1 if peer_pressure == "Yes" else 0

    # Row 4: Allergy, Wheezing, Fatigue
    col1, col2, col3 = st.columns(3)
    with col1:
        allergy = st.selectbox("Allergy", ["Yes", "No"])
        allergy = 1 if allergy == "Yes" else 0
    with col2:
        wheezing = st.selectbox("Wheezing", ["Yes", "No"])
        wheezing = 1 if wheezing == "Yes" else 0
    with col3:
        fatigue = st.selectbox("Fatigue", ["Yes", "No"])
        fatigue = 1 if fatigue == "Yes" else 0

    # Row 5: Coughing, Shortness of Breath, Swallowing Difficulty
    col1, col2, col3 = st.columns(3)
    with col1:
        coughing = st.selectbox("Coughing", ["Yes", "No"])
        coughing = 1 if coughing == "Yes" else 0
    with col2:
        shortness_of_breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
        shortness_of_breath = 1 if shortness_of_breath == "Yes" else 0
    with col3:
        swallowing_difficulty = st.selectbox("Difficulty Swallowing", ["Yes", "No"])
        swallowing_difficulty = 1 if swallowing_difficulty == "Yes" else 0

    # Combine all inputs into a single list
    input_data = [
        gender, age, smoking, yellow_fingers, anxiety, chest_pain,
        chronic_disease, peer_pressure, fatigue, allergy, wheezing,
        alcohol, coughing, shortness_of_breath, swallowing_difficulty
    ]

    # Predict using the loaded model
    if st.button("Predict"):
        prediction = model.predict([input_data])
        if prediction[0] == 1:
            result = '<span style="color:green;">Positive for Lung Cancer</span>'
        else:
            result = '<span style="color:red;">Negative for Lung Cancer</span>'
        
        st.markdown(f"### Prediction: {result}", unsafe_allow_html=True)


if __name__ == "__main__":
    page = st.sidebar.radio("Select a Sampling Technique or Predictor", ["K-Fold Cross-Validation(KFOLD)", "Leave One-Out Cross Validation (LOOCV)", "Lung Cancer Diagnosis Predictor"])
    if page == "K-Fold Cross-Validation(KFOLD)":
        kfold_cv()
    elif page == "Leave One-Out Cross Validation (LOOCV)":
        loocv()
    elif page == "Lung Cancer Diagnosis Predictor":
        predictor()
