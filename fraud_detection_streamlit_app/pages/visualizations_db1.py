import streamlit as st
import pandas as pd
import folium
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from utils.data_loader import fetch_data_from_hbase
from streamlit.components.v1 import html

@st.cache_resource
def run():
    # Fetch data for Database 1
    df = fetch_data_from_hbase(database="Database 1", limit=100000)

    # Convert 'trans_date_trans_time' to datetime with the correct format
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format="%d/%m/%Y %H:%M")
    
    # Convert 'dob' to datetime (if it is in the same format)
    df['dob'] = pd.to_datetime(df['dob'], format="%d/%m/%Y")

    # Extract the hour of the transaction from 'trans_date_trans_time'
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    
    # Calculate age in days from 'dob' and 'trans_date_trans_time'
    df['age_days'] = (df['trans_date_trans_time'] - df['dob']).dt.days

    # Step 1: Filter fraudulent transactions
    fraud_transactions = df[df['is_fraud'] == 1]

    # Step 2: Feature Engineering
    # Encoding the 'category' column
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])

    # Define features (including lat, long)
    X = df[['amt', 'age_days', 'trans_hour', 'city_pop', 'lat', 'long', 'category_encoded']]

    # Step 3: Convert columns to numeric, handling errors
    X['amt'] = pd.to_numeric(X['amt'], errors='coerce')
    X['city_pop'] = pd.to_numeric(X['city_pop'], errors='coerce')
    X['lat'] = pd.to_numeric(X['lat'], errors='coerce')
    X['long'] = pd.to_numeric(X['long'], errors='coerce')

    # Handle missing values by filling NaNs with 0 or another strategy
    X.fillna(0, inplace=True)  # Optionally, you can replace with the mean or median instead of 0

    # Ensure the target 'is_fraud' is integer (0 and 1)
    df['is_fraud'] = df['is_fraud'].astype(int)

    y = df['is_fraud']

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 5: Model Training (XGBoost)
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    # Step 6: Model Evaluation
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]  # Get probabilities for ROC curve and AUC

    # Accuracy and ROC AUC Score
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    st.write("\nClassification Report:")
    st.text(classification_report(y_test, y_pred))

    # Step 7: Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Step 8: ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)

    # Step 9: Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    st.pyplot(fig)

    # Step 10: Cross-validation (k-fold)
    cv_scores = cross_val_score(xgb_model, X, y, cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True), scoring='accuracy')
    st.write(f"Cross-validated accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
     #MAP
    # Step 1: Filter fraudulent transactions
    fraud_transactions = df[df['is_fraud'] == 1]

    # Ensure that the 'lat' and 'long' columns are numeric
    fraud_transactions['lat'] = pd.to_numeric(fraud_transactions['lat'], errors='coerce')
    fraud_transactions['long'] = pd.to_numeric(fraud_transactions['long'], errors='coerce')

    # Drop rows where 'lat' or 'long' is NaN after conversion
    fraud_transactions = fraud_transactions.dropna(subset=['lat', 'long'])

    # Step 2: Create a base map centered on a specific location (e.g., the average latitude and longitude)
    map_center = [fraud_transactions['lat'].mean(), fraud_transactions['long'].mean()]
    m = folium.Map(location=map_center, zoom_start=2)

    # Step 3: Add markers for fraudulent transactions to the map
    for idx, row in fraud_transactions.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6
        ).add_to(m)

    # Step 4: Display the map directly in Streamlit using the HTML embed method
    map_html = m._repr_html_()  # Get the map HTML representation
    html(map_html, height=600)  # Embed the HTML into Streamlit


