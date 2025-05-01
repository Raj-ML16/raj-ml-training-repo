import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import streamlit as st

# Function to load Iris dataset
def load_data():
    try:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])
        st.success("Dataset loaded successfully.")
        return df, iris.feature_names, iris.target_names
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None, None

# Module 1: Exploratory Data Analysis
def perform_eda(df, feature_names, target_names):
    st.header("Exploratory Data Analysis")

    # Basic statistics
    st.subheader("Dataset Statistics")
    st.write(df[feature_names].describe())

    # Class distribution
    st.subheader("Class Distribution")
    st.write(df['target_name'].value_counts())

    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Correlation matrix
    st.subheader("Correlation Matrix")
    try:
        corr_matrix = df[feature_names].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering correlation matrix: {e}")

    # Pairplot
    st.subheader("Pairplot of Features")
    try:
        fig = sns.pairplot(df, hue='target_name', vars=feature_names)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering pairplot: {e}")

    # Boxplots
    st.subheader("Boxplots for Outlier Detection")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        df[feature_names].boxplot(ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering boxplots: {e}")

# Module 2: Model Training and Evaluation
def train_and_evaluate_models(df, feature_names, target_names):
    st.header("Model Training & Evaluation")

    # Feature selection
    st.subheader("Select Features for Training")
    selected_features = st.multiselect("Choose features", feature_names, default=feature_names)
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        return None, None

    # Data preparation
    try:
        X = df[selected_features]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None, None

    # Model selection
    model_type = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])

    # Train model
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=200)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    try:
        model.fit(X_train, y_train)
        st.success(f"{model_type} trained successfully.")
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None

    # Predictions
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None, None

    # Evaluation
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Cross-validation
    try:
        cv_scores = cross_val_score(model, X, y, cv=5)
        st.write(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})")
    except Exception as e:
        st.error(f"Error in cross-validation: {e}")

    # Classification report
    st.subheader("Classification Report")
    try:
        st.text(classification_report(y_test, y_pred, target_names=target_names))
    except Exception as e:
        st.error(f"Error rendering classification report: {e}")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    try:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering confusion matrix: {e}")

    # Feature importance
    if model_type == "Logistic Regression" and len(selected_features) == len(feature_names):
        st.subheader("Feature Importance (Logistic Regression Coefficients)")
        try:
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': np.abs(model.coef_).mean(axis=0)
            }).sort_values(by='Coefficient', ascending=False)
            st.write(coef_df)
        except Exception as e:
            st.error(f"Error rendering feature importance: {e}")
    elif model_type == "Random Forest":
        st.subheader("Feature Importance (Random Forest)")
        try:
            importances = pd.DataFrame({
                'Feature': selected_features,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            st.write(importances)
        except Exception as e:
            st.error(f"Error rendering feature importance: {e}")

    return model, selected_features

# Module 3: Predict Iris Species
def predict_iris_species(df, model, feature_names, selected_features, target_names):
    st.header("Predict Iris Species")
    st.markdown("Enter feature values to predict the Iris species")

    # Input sliders for selected features
    inputs = {}
    for feature in selected_features:
        try:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            inputs[feature] = st.slider(f"{feature}", min_val, max_val, mean_val)
        except Exception as e:
            st.error(f"Error creating slider for {feature}: {e}")
            return

    # Prediction
    if st.button("Predict"):
        try:
            input_data = pd.DataFrame([inputs])
            prediction = model.predict(input_data)[0]
            st.write(f"Predicted Iris Species: **{target_names[prediction]}**")
            st.success("Prediction successful.")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Main script
def main():
    # Streamlit styling (no background image)
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
            color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .stSlider>div>div>div {
            background-color: #4CAF50;
        }
        .stSelectbox, .stMultiselect {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        h1, h2, h3 {
            color: #4CAF50;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Iris Species Classification Dashboard 🌸")

    # Load dataset
    df, feature_names, target_names = load_data()
    if df is None:
        st.error("Failed to load dataset. Please check dependencies and try again.")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    module = st.sidebar.selectbox("Select Module", [
        "Exploratory Data Analysis",
        "Model Training & Evaluation",
        "Predict Iris Species"
    ], index=0)  # Default to EDA to ensure initial rendering

    # Conditional display based on selected module
    if module == "Exploratory Data Analysis":
        perform_eda(df, feature_names, target_names)
    elif module == "Model Training & Evaluation":
        model, selected_features = train_and_evaluate_models(df, feature_names, target_names)
        if model is not None:
            st.session_state.model = model
            st.session_state.selected_features = selected_features
    elif module == "Predict Iris Species":
        if hasattr(st.session_state, 'model') and hasattr(st.session_state, 'selected_features'):
            predict_iris_species(df, st.session_state.model, feature_names, st.session_state.selected_features, target_names)
        else:
            st.error("Please run the Model Training & Evaluation module first to train the model.")

    # Warning for prediction
    if module == "Predict Iris Species":
        st.warning("Note: Predictions are based on the trained model. Ensure input values are realistic.")

if __name__ == "__main__":
    main()