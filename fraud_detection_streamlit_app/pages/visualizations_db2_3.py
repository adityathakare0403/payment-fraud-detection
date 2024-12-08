import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

from utils.data_loader import fetch_data_from_hbase  # Ensure proper relative import

# Function to run visualization logic
@st.cache_resource
def run():
    st.title("Machine Learning Analysis - Fraud Detection (Visualization 3)")
    st.write("### Using Decision Tree Classifier for Customer Age Focused Features")

    # Fetch and preprocess the data
    df = fetch_data_from_hbase(database="Database 2", limit=100000)

    # Ensure necessary columns are numeric
    df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
    df['customer_age'] = pd.to_numeric(df['customer_age'], errors='coerce')
    df['account_age_days'] = pd.to_numeric(df['account_age_days'], errors='coerce')

    # Drop missing values for simplicity
    df = df.dropna(subset=['customer_age', 'transaction_amount', 'payment_method', 'account_age_days', 'is_fraudulent'])

    # Step 1: Data Preparation - Focus on Customer Age and relevant features
    X = df[['customer_age', 'transaction_amount', 'payment_method', 'account_age_days']]
    y = df['is_fraudulent']

    # Convert 'payment_method' to dummy variables (One-Hot Encoding)
    X = pd.get_dummies(X, columns=['payment_method'], drop_first=True)

    # Apply SMOTE to balance the dataset (oversample minority class)
    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Step 5: Model 1 - Decision Tree Classifier
    st.write("### Decision Tree Classifier")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    # Display classification report for Decision Tree
    st.text("Decision Tree Classification Report:")
    st.text(classification_report(y_test, y_pred_dt))


    # Confusion Matrix for Decision Tree
    st.write("### Confusion Matrix for Decision Tree")
    ConfusionMatrixDisplay.from_estimator(dt_model, X_test, y_test, display_labels=["Not Fraud", "Fraud"], cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix - Decision Tree")
    st.pyplot(plt)
