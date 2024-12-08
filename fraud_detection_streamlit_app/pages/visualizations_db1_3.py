import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utils.data_loader import fetch_data_from_hbase  # Assuming you are using this to fetch data

# Function to run the visualization logic
@st.cache_resource
def run():
    st.title("Fraudulent Transactions Analysis - Database 1")
    st.write("### Using Random Forest Classifier")

    # Fetch and preprocess the data from HBase
    df = fetch_data_from_hbase(database="Database 1", limit=10000)

    # Feature engineering
    df['amt'] = pd.to_numeric(df['amt'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['long'] = pd.to_numeric(df['long'], errors='coerce')
    df['merch_lat'] = pd.to_numeric(df['merch_lat'], errors='coerce')
    df['merch_long'] = pd.to_numeric(df['merch_long'], errors='coerce')
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%d/%m/%Y %H:%M')
    df['dob'] = pd.to_datetime(df['dob'], format='%d/%m/%Y')

    # Check that 'trans_date_trans_time' and 'dob' exist
    if 'trans_date_trans_time' in df.columns and 'dob' in df.columns:
        # Calculate 'trans_hour' and 'age_days'
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['age_days'] = (df['trans_date_trans_time'] - df['dob']).dt.days
    else:
        st.error("Required columns 'trans_date_trans_time' or 'dob' are missing in the dataset.")

    # Ensure the new columns are created successfully
    st.write(f"Columns after feature engineering: {df.columns.tolist()}")

    # Encode the 'category' column
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])

    # Select features and target variable
    try:
        X = df[['category_encoded', 'amt', 'age_days', 'trans_hour', 'city_pop']]
    except KeyError as e:
        st.error(f"KeyError: The column(s) {e.args[0]} are missing.")
        return

    y = df['is_fraud']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)

    # Display classification report
    st.subheader("Model Evaluation - Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    st.subheader("Confusion Matrix")
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)
