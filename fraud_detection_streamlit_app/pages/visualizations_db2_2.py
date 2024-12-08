import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

from utils.data_loader import fetch_data_from_hbase  # Ensure proper relative import

# Function to run visualization logic
@st.cache_resource
def run():
    st.title("Machine Learning Analysis - Fraud Detection (Visualization 2)")
    st.write("### Using Decision Tree and CatBoost Classifiers")

    # Fetch and preprocess the data
    df = fetch_data_from_hbase(database="Database 2", limit=10000)

    # Ensure necessary columns are numeric
    df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
    df['customer_age'] = pd.to_numeric(df['customer_age'], errors='coerce')
    df['account_age_days'] = pd.to_numeric(df['account_age_days'], errors='coerce')

    # Drop missing values for simplicity
    df = df.dropna(subset=['customer_age', 'transaction_amount', 'payment_method', 'account_age_days', 'is_fraudulent'])

    # Prepare features (X) and target (y)
    X = df[['transaction_hour', 'transaction_amount', 'account_age_days', 'customer_age', 'payment_method']]
    y = df['is_fraudulent']

    # Convert categorical variable 'payment_method' to dummy variables
    X = pd.get_dummies(X, columns=['payment_method'], drop_first=True)

    # Balance the dataset using SMOTE
    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    ### Model 1 - Decision Tree Classifier
    st.write("### Decision Tree Classifier")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    # Display classification report for Decision Tree
    st.text("Decision Tree Classification Report:")
    st.text(classification_report(y_test, y_pred_dt))

    ### Model 2 - CatBoost Classifier
    st.write("### CatBoost Classifier")
    catboost_model = CatBoostClassifier(iterations=1000, depth=10, learning_rate=0.1, random_state=42, verbose=0)
    catboost_model.fit(X_train, y_train)
    y_pred_catboost = catboost_model.predict(X_test)

    # Display classification report for CatBoost
    st.text("CatBoost Classification Report:")
    st.text(classification_report(y_test, y_pred_catboost))

    ### Visualizations

    # Confusion Matrix for Decision Tree
    st.write("### Confusion Matrix for Decision Tree")
    ConfusionMatrixDisplay.from_estimator(dt_model, X_test, y_test, display_labels=["Not Fraud", "Fraud"], cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix - Decision Tree")
    st.pyplot(plt)

    # Confusion Matrix for CatBoost
    st.write("### Confusion Matrix for CatBoost")
    ConfusionMatrixDisplay.from_estimator(catboost_model, X_test, y_test, display_labels=["Not Fraud", "Fraud"], cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix - CatBoost")
    st.pyplot(plt)

    # Feature Importance for CatBoost
    st.write("### Feature Importance for CatBoost Classifier")
    feature_importances = pd.Series(catboost_model.feature_importances_, index=X_train.columns)
    plt.figure(figsize=(8, 6))
    feature_importances.sort_values().plot(kind="barh", color="teal")
    plt.title("Feature Importance in Predicting Fraudulent Transactions (CatBoost)")
    plt.xlabel("Feature Importance Score")
    st.pyplot(plt)
