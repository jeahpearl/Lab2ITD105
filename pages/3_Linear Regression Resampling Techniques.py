import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Predefined weather dataset path
WEATHER_CSV_PATH = r"C:\Users\jeahp\Desktop\ITD105\Lab2\csv\weather.csv"

# Set the title of the app
st.title("Linear Regression Sampling Techniques")

# Navigation sidebar
selection = st.sidebar.radio("Select a Sampling Technique or Predictor", 
                             ["Split Into Train and Test Sets", "Repeated Random Train-Test Splits", "Temperature Predictor"])

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    # Load the dataset
    dataframe = pd.read_csv(WEATHER_CSV_PATH)
    
    # Show original data preview
    st.subheader("Dataset Preview")
    with st.expander("Raw Dataset"):
        st.dataframe(dataframe.head())
    
    # Encode 'Description' column to numeric values
    label_encoder = LabelEncoder()
    dataframe['Description'] = label_encoder.fit_transform(dataframe['Description'])
    
    # Preprocessed data preview
    st.subheader("Preprocessed Data Preview")
    with st.expander("Preprocessed Dataset"):
        st.dataframe(dataframe.head())
    
    return dataframe

# 1. Split Into Train and Test Sets
if selection == "Split Into Train and Test Sets":
    st.header("1. Train Model with Split into Train and Test Sets")

    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Features (X) and target (Y)
    X = df.drop(columns=["Temperature_c"])  # All columns except Temperature_c
    Y = df["Temperature_c"]                # Target column
    
    # Train-test split
    test_size = st.slider("Test size (as a percentage)", 10, 50, 20) / 100
    seed = 42
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # Train the model
    model = RandomForestRegressor(random_state=seed)
    model.fit(X_train, Y_train)

    # Calculate R² on test data
    r2_score = model.score(X_test, Y_test)
    st.subheader("Model Performance (Test Set)")
    st.write(f"Mean R²: {r2_score:.3f}")
    
    # Option to download the model
    if st.button("Download the Trained Model"):
        model_folder = r"C:\Users\jeahp\Desktop\ITD105\Lab2\Models"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_filename = os.path.join(model_folder, "SITTS_weather_model.joblib")
        joblib.dump(model, model_filename)
        st.success(f"Model trained and saved as {model_filename}")

# 2. Repeated Random Train-Test Splits
elif selection == "Repeated Random Train-Test Splits":
    st.header("2. Train Model with Repeated Random Train-Test Splits")

    # Load and preprocess data
    df = load_and_preprocess_data()

    # Features (X) and target (Y)
    X = df.drop(columns=["Temperature_c"])  # All columns except Temperature_c
    Y = df["Temperature_c"]                # Target column
    
    # Parameters for repeated random splits
    n_splits = st.slider("Number of splits:", 2, 20, 10)
    test_size = st.slider("Test size proportion:", 0.1, 0.5, 0.33)
    seed = 42

    # Perform Repeated Random Test-Train Splits
    shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    model = LinearRegression()
    results = cross_val_score(model, X, Y, cv=shuffle_split, scoring="r2")
    
    # Display results
    st.subheader("Repeated Random Test-Train Splits Results")
    st.write(f"Mean R²: {results.mean():.3f}")
    st.write(f"R² Standard Deviation: {results.std():.3f}")
    
    # Train the model and provide download option
    if st.button("Train and Download Model"):
        model.fit(X, Y)
        model_folder = r"C:\Users\jeahp\Desktop\ITD105\Lab2\Models"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_filename = os.path.join(model_folder, "RRTTS_weather_model.joblib")
        joblib.dump(model, model_filename)
        st.success(f"Model trained and saved as {model_filename}")

# 3. Temperature Predictor
elif selection == "Temperature Predictor":
    st.header("Temperature Predictor")

    # Predefined model file paths
    MODEL_PATHS = {
        "Repeated Random Train-Test Splits Model (RRTTS)": r"C:\Users\jeahp\Desktop\ITD105\Lab2\Models\RRTTS_weather_model.joblib",
        "Split Into Train-Test Sets Model (SITTS)": r"C:\Users\jeahp\Desktop\ITD105\Lab2\Models\SITTS_weather_model.joblib"
    }

    # Model selection dropdown
    st.subheader("Select a Preloaded Model")
    model_choice = st.selectbox("Choose a model:", list(MODEL_PATHS.keys()))

    # Verify if the model file exists
    model_path = MODEL_PATHS[model_choice]
    if not os.path.exists(model_path):
        st.error(f"Error: The model file '{model_path}' does not exist. Please ensure the file is available.")
    else:
        try:
            # Load the selected model
            model = joblib.load(model_path)
            st.success(f"Model '{model_choice}' loaded successfully!")

            # Input form for prediction
            st.subheader("Input Weather Data for Prediction")
            col1, col2, col3 = st.columns(3)
            with col1:
                humidity = st.number_input(
                    "What is the current humidity level? (0.0 to 1.0)", 
                    value=0.5, min_value=0.0, max_value=1.0, step=0.01
                )
                wind_speed = st.number_input(
                    "How fast is the wind blowing? (km/h)", 
                    value=10.0, min_value=0.0, max_value=53.24, step=0.1
                )

            with col2:
                wind_bearing = st.number_input(
                    "What is the wind bearing or direction? (in degrees)", 
                    value=180, min_value=0, max_value=360, step=1
                )
                visibility = st.number_input(
                    "How far can you see? (Visibility in km)", 
                    value=5.0, min_value=0.0, max_value=16.1, step=0.1
                )

            with col3:
                pressure = st.number_input(
                    "What is the atmospheric pressure? (in millibars)", 
                    value=1010.0, min_value=900.0, max_value=1100.0, step=0.1
                )
                rain = st.selectbox(
                    "Is it currently raining?", 
                    ["No", "Yes"]
                )
                rain_encoded = {"No": 0, "Yes": 1}[rain]

            # Weather description input
            description = st.selectbox(
                "How would you describe the weather?", 
                ["Cold", "Normal", "Warm"]
            )
            description_encoded = {"Cold": 0, "Normal": 1, "Warm": 2}[description]


            # Combine inputs into a single array
            input_data = np.array([[humidity, wind_speed, wind_bearing, visibility, pressure, rain_encoded, description_encoded]])

            # Predict
            if st.button("Predict Temperature"):
                prediction = model.predict(input_data)
                st.success(f"Predicted Temperature: {prediction[0]:.2f} °C")

        except Exception as e:
            st.error(f"Error during model loading or prediction: {e}")
