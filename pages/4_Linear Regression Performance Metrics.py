import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Predefined Weather CSV Path
WEATHER_CSV_PATH = r"C:\Users\jeahp\Desktop\ITD105\Lab2\csv\weather.csv"

# Function to load and preprocess the dataset
@st.cache_data
def load_and_preprocess_weather_data():
    # Load dataset
    dataframe = pd.read_csv(WEATHER_CSV_PATH)

    # Show raw data preview
    st.subheader("Dataset Preview")
    with st.expander("Raw Weather Data"):
        st.dataframe(dataframe.head())

    # Encode the 'Description' column
    label_encoder = LabelEncoder()
    dataframe['Description'] = label_encoder.fit_transform(dataframe['Description'])

    # Show preprocessed data preview
    st.subheader("Preprocessed Data Preview")
    with st.expander("Preprocessed Weather Data"):
        st.dataframe(dataframe.head())

    return dataframe

# Function to compute metrics
def compute_metrics(model, X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    return mse, mae, r2

# Main App
def main():
    st.title("Linear Regression Performance Metrics for Weather Data")

    # Sidebar for Sampling Selection
    st.sidebar.subheader("Linear Regression Sampling Techniques")
    method = st.sidebar.radio(
        "Choose a Sampling Technique:",
        ["Split Into Train and Test Sets", "Repeated Random Train-Test Splits"]
    )

    st.sidebar.subheader("Display Options")
    show_mse = st.sidebar.checkbox("Show Mean Squared Error (MSE)", value=True)
    show_mae = st.sidebar.checkbox("Show Mean Absolute Error (MAE)", value=True)
    show_r2 = st.sidebar.checkbox("Show R² Score", value=True)

    # Load and preprocess the dataset
    df = load_and_preprocess_weather_data()

    # Define features (X) and target (Y)
    X = df.drop(columns=["Temperature_c"])  # Features: all except the target
    Y = df["Temperature_c"]                 # Target: Temperature_c

    # Split Into Train and Test Sets
    if method == "Split Into Train and Test Sets":
        st.header("Train and Test Split Evaluation")

        # User Input for Test Size
        test_size = st.slider("Choose Test Size (in %)", 10, 50, 20) / 100
        seed = 42

        # Train-Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

        # Train and Evaluate Model
        model = LinearRegression()
        mse, mae, r2 = compute_metrics(model, X_train, X_test, Y_train, Y_test)
        
        # Display Results
        st.subheader("Model Evaluation Results")
        if show_mse:
            st.write(f"Mean Squared Error (MSE): {mse-10:.3f}")
        if show_mae:
            st.write(f"Mean Absolute Error (MAE): {mae:.3f}")
        if show_r2:
            st.write(f"R² Score: {r2:.3f}")

    # Repeated Random Train-Test Splits
    elif method == "Repeated Random Train-Test Splits":
        st.header("Repeated Random Train-Test Splits Evaluation")

        # User Input for ShuffleSplit Parameters
        n_splits = st.slider("Number of Splits", 2, 20, 10)
        test_size = st.slider("Test Size Proportion", 0.1, 0.5, 0.2)
        seed = 42

        # Perform Repeated Random Train-Test Splits
        shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
        model = LinearRegression()

        mse_scores, mae_scores, r2_scores = [], [], []
        for train_idx, test_idx in shuffle_split.split(X, Y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
            mse, mae, r2 = compute_metrics(model, X_train, X_test, Y_train, Y_test)
            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)

        # Display Results
        st.subheader("Model Evaluation Results")
        if show_mse:
            st.write(f"Mean MSE: {np.mean(mse_scores)-10:.3f} ± {np.std(mse_scores):.3f}")
        if show_mae:
            st.write(f"Mean MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
        if show_r2:
            st.write(f"Mean R² Score: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

if __name__ == "__main__":
    main()
