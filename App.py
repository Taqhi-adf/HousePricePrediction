
#📊 Streamlit App: House Price Prediction (Multiple Regression Models)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction – Regression Models Comparison")

# -----------------------------
# Load Dataset
# -----------------------------
st.header("1️⃣ Load Dataset")

uploaded_file = st.file_uploader(
    "Upload Cleaned House Price CSV file",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=0)
    st.success("Dataset loaded successfully!")

    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    # -----------------------------
    # Data Cleaning
    # -----------------------------
    st.header("2️⃣ Data Cleaning & Preprocessing")

    st.write("Missing values before cleaning:")
    st.write(df.isnull().sum())

    df['House_type'].fillna(method='bfill', inplace=True)
    df['Garage'].fillna(method='bfill', inplace=True)

    df['Year_built'] = df['Year_built'].astype('int64')

    df.drop(columns=['City', 'House_type', 'Garage'], inplace=True)

    st.write("Missing values after cleaning:")
    st.write(df.isnull().sum())

    st.subheader("Dataset Info")
    st.text(df.info())

    # -----------------------------
    # Feature Selection
    # -----------------------------
    st.header("3️⃣ Feature Selection")

    X = df[['Year_built', 'Area_in_Sqft', 'Bedrooms',
            'Bathrooms', 'Listing_Date', 'House_Age']]
    y = df['Price']

    st.write("Selected Features:")
    st.write(X.head())

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    st.header("4️⃣ Train-Test Split")

    test_size = st.slider("Select Test Size (%)", 10, 40, 20) / 100

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    st.write(f"Training samples: {X_train.shape[0]}")
    st.write(f"Testing samples: {X_test.shape[0]}")

    # -----------------------------
    # Train Models
    # -----------------------------
    st.header("5️⃣ Model Training & Evaluation")

    results = []

    def evaluate_model(name, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results.append([name, mse, mae, rmse, r2])

    # Linear Regression
    m1 = LinearRegression()
    m1.fit(X_train, y_train)
    y_pred1 = m1.predict(X_test)
    evaluate_model("Linear Regression", y_test, y_pred1)

    # Decision Tree
    m2 = DecisionTreeRegressor(criterion='squared_error', max_depth=5)
    m2.fit(X_train, y_train)
    y_pred2 = m2.predict(X_test)
    evaluate_model("Decision Tree Regressor", y_test, y_pred2)

    # Random Forest
    m3 = RandomForestRegressor(n_estimators=100, random_state=42)
    m3.fit(X_train, y_train)
    y_pred3 = m3.predict(X_test)
    evaluate_model("Random Forest Regressor", y_test, y_pred3)

    # XGBoost
    m4 = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    m4.fit(X_train, y_train)
    y_pred4 = m4.predict(X_test)
    evaluate_model("XGBoost Regressor", y_test, y_pred4)

    # -----------------------------
    # Results Comparison
    # -----------------------------
    st.header("6️⃣ Model Comparison")

    df_models = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Mean Squared Error",
            "Mean Absolute Error",
            "Root Mean Squared Error",
            "R2 Score"
        ]
    )

    st.dataframe(df_models)

    best_model = df_models.sort_values("R2 Score", ascending=False).iloc[0]

    st.success(
        f"🏆 Best Model: **{best_model['Model']}** "
        f"with R² Score = **{best_model['R2 Score']:.3f}**"
    )

    # -----------------------------
    # Visualization
    # -----------------------------
    st.header("7️⃣ R² Score Comparison")

    fig, ax = plt.subplots()
    ax.bar(df_models["Model"], df_models["R2 Score"])
    ax.set_ylabel("R² Score")
    ax.set_title("Model Performance Comparison")
    plt.xticks(rotation=30)

    st.pyplot(fig)

else:
    st.info("👆 Please upload the CSV file to start.")
