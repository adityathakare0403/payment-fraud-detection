import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import seaborn as sns


from utils.data_loader import fetch_data_from_hbase  # Import the data fetching function

# Haversine function for calculating distance between two geographical points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Function to run the visualization logic
@st.cache_resource
def run():
    st.title("Fraudulent Transactions Analysis - Database 1")
    st.write("### Using Gradient Boosting Classifier")

    # Fetch and preprocess the data from HBase
    df = fetch_data_from_hbase(database="Database 1", limit=10000)

    # Convert relevant columns to numeric values
    df['amt'] = pd.to_numeric(df['amt'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['long'] = pd.to_numeric(df['long'], errors='coerce')
    df['merch_lat'] = pd.to_numeric(df['merch_lat'], errors='coerce')
    df['merch_long'] = pd.to_numeric(df['merch_long'], errors='coerce')
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%d/%m/%Y %H:%M')
    df['dob'] = pd.to_datetime(df['dob'], format='%d/%m/%Y')

    # Feature engineering
    df['distance'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_day'] = df['trans_date_trans_time'].dt.day
    df['age_days'] = (df['trans_date_trans_time'] - df['dob']).dt.days

    # Encode categorical variables
    categorical_columns = ['merchant', 'category', 'gender', 'job']
    label_encoders = {col: LabelEncoder() for col in categorical_columns}

    for col, encoder in label_encoders.items():
        df[col] = encoder.fit_transform(df[col])

    # Select features and target variable
    features = ['amt', 'distance', 'city_pop', 'trans_hour', 'trans_day', 'age_days'] + categorical_columns
    target = 'is_fraud'
    X = df[features]
    y = df[target]

    # Balance the dataset using SMOTE
    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Train a Gradient Boosting model
    gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100)
    gb_model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = gb_model.predict(X_test)
    y_pred_prob = gb_model.predict_proba(X_test)[:, 1]

    # Display classification report
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Display ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    st.write(f"### ROC-AUC Score: {roc_auc:.4f}")

    # Plot confusion matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 16}, xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)



