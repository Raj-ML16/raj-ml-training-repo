import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

# Load dataset
df = pd.read_csv('cars.csv')

# EDA
st.title("Car Weight Prediction Dashboard 🚗")
st.header("Exploratory Data Analysis")

# Basic statistics
st.subheader("Dataset Statistics")
st.write(df.describe())

# Correlation analysis
st.subheader("Correlation Matrix")
corr_matrix = df.corr()
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Pairplot
st.subheader("Pairplot of Variables")
fig = sns.pairplot(df)
st.pyplot(fig)

# Check for missing values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# VIF calculation
st.subheader("Variance Inflation Factor (VIF)")
X_vif = df[['HP', 'MPG', 'VOL', 'SP']].copy()
vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
st.write(vif_data)

# Data preparation
X_slr = df[['HP']]
X_mlr = df[['HP', 'MPG', 'VOL', 'SP']]
y = df['WT']

# Train-test split
X_slr_train, X_slr_test, y_train, y_test = train_test_split(X_slr, y, test_size=0.3, random_state=42)
X_mlr_train, X_mlr_test, y_train, y_test = train_test_split(X_mlr, y, test_size=0.3, random_state=42)

# Simple Linear Regression
slr_model = LinearRegression()
slr_model.fit(X_slr_train, y_train)
slr_pred = slr_model.predict(X_slr_test)

# Multiple Linear Regression
mlr_model = LinearRegression()
mlr_model.fit(X_mlr_train, y_train)
mlr_pred = mlr_model.predict(X_mlr_test)

# Model evaluation
st.header("Model Performance")

# SLR Results
st.subheader("Simple Linear Regression (WT vs HP)")
st.write(f"Coefficient: {slr_model.coef_[0]:.4f}")
st.write(f"Intercept: {slr_model.intercept_:.4f}")
st.write(f"R² Score: {r2_score(y_test, slr_pred):.4f}")
st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, slr_pred)):.4f}")

# MLR Results
st.subheader("Multiple Linear Regression (WT vs All Variables)")
st.write("Coefficients:")
for var, coef in zip(X_mlr.columns, mlr_model.coef_):
    st.write(f"{var}: {coef:.4f}")
st.write(f"Intercept: {mlr_model.intercept_:.4f}")
st.write(f"R² Score: {r2_score(y_test, mlr_pred):.4f}")
st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, mlr_pred)):.4f}")

# Streamlit prediction interface
st.header("Predict Car Weight")
st.markdown("Enter values to predict car weight using MLR model")

hp = st.slider("Horsepower (HP)", float(df['HP'].min()), float(df['HP'].max()), float(df['HP'].mean()))
mpg = st.slider("Miles Per Gallon (MPG)", float(df['MPG'].min()), float(df['MPG'].max()), float(df['MPG'].mean()))
vol = st.slider("Volume (VOL)", float(df['VOL'].min()), float(df['VOL'].max()), float(df['VOL'].mean()))
sp = st.slider("Speed (SP)", float(df['SP'].min()), float(df['SP'].max()), float(df['SP'].mean()))

# Prediction
input_data = np.array([[hp, mpg, vol, sp]])
pred_weight = mlr_model.predict(input_data)[0]
st.write(f"Predicted Car Weight: {pred_weight:.2f} units")

# Streamlit beautification
st.markdown("""
    <style>
    .main {
        background-image: url('https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        color: white;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
    }
    .stSlider>div>div>div {
        background-color: #FF4B4B;
    }
    h1, h2, h3 {
        color: #FF4B4B;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# Warning for prediction
st.warning("Note: Predictions are based on the trained MLR model. Ensure input values are realistic.")