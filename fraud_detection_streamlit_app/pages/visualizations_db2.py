import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loader import fetch_data_from_hbase  # Ensure proper relative import

# Function to run visualization logic
@st.cache_resource
def run():
    st.title("Machine Learning Analysis - Fraud Detection")
    st.write("### Using k-Nearest Neighbors and Gradient Boosting Classifiers")

    # Fetch and preprocess the data
    df = fetch_data_from_hbase(database="Database 2", limit=10000)

    # Ensure necessary columns are numeric
    df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
    df['customer_age'] = pd.to_numeric(df['customer_age'], errors='coerce')
    df['account_age_days'] = pd.to_numeric(df['account_age_days'], errors='coerce')

    # Drop missing values for simplicity
    df = df.dropna(subset=['customer_age', 'transaction_amount', 'payment_method', 'account_age_days', 'is_fraudulent'])

    # Prepare features (X) and target (y)
    X = df[['customer_age', 'transaction_amount', 'payment_method', 'account_age_days']]
    y = df['is_fraudulent']

    # Convert categorical variable 'payment_method' to dummy variables
    X = pd.get_dummies(X, columns=['payment_method'], drop_first=True)

    # Balance the dataset using SMOTE
    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    ### Model 1 - k-Nearest Neighbors
    st.write("### k-Nearest Neighbors Classifier")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    # Display classification report for k-NN
    st.text("k-Nearest Neighbors Classification Report:")
    st.text(classification_report(y_test, y_pred_knn))

    ### Model 2 - Gradient Boosting
    st.write("### Gradient Boosting Classifier")
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)

    # Display classification report for Gradient Boosting
    st.text("Gradient Boosting Classification Report:")
    st.text(classification_report(y_test, y_pred_gb))

    ### Visualizations
    # Plot ROC Curves
    st.write("### ROC Curves for Models")
    def plot_roc_curve(model, X_test, y_test, label):
        # Ensure y_test is numeric and pass pos_label explicitly to handle binary classification
        y_test = y_test.astype(int)

        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate ROC curve and AUC, with pos_label=1 for fraud detection
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=1)
        auc_score = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f"{label} AUC = {auc_score:.2f}")

    plt.figure(figsize=(10, 6))
    plot_roc_curve(knn_model, X_test, y_test, "k-Nearest Neighbors")
    plot_roc_curve(gb_model, X_test, y_test, "Gradient Boosting")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    st.pyplot(plt)

    # Feature Importance for Gradient Boosting
    st.write("### Feature Importance for Gradient Boosting Classifier")
    feature_importances = pd.Series(gb_model.feature_importances_, index=X_train.columns)
    plt.figure(figsize=(8, 6))
    feature_importances.sort_values().plot(kind="barh", color="teal")
    plt.title("Feature Importance in Predicting Fraudulent Transactions")
    plt.xlabel("Feature Importance Score")
    st.pyplot(plt)
