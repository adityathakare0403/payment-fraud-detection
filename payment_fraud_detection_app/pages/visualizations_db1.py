import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE

def run():
    st.title("📈 Visualizations")

    # Load updated data from session state
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df.copy()

        # Display Visualizations
        st.write("### Visualizations: Fraudulent Transactions Analysis")
        
        # Filter fraudulent transactions
        fraud_data = df[df['is_fraudulent'] == "1"]
        if fraud_data.empty:
            st.warning("No fraudulent transactions available for visualization.")
        else:
            st.write("#### Distribution of Transaction Amounts in Fraudulent Transactions by Payment Method")
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='payment_method', y='transaction_amount', data=fraud_data)
            plt.title('Transaction Amounts in Fraudulent Transactions by Payment Method')
            plt.ylabel('Transaction Amount')
            plt.xlabel('Payment Method')
            plt.xticks(rotation=45)
            st.pyplot(plt)

        # Machine Learning Section
        st.write("### Fraud Prediction Using Machine Learning Models")
        
        try:
            dfa1 = df.copy()
            dfa1['is_fraudulent'] = dfa1['is_fraudulent'].astype(int)
            
            X = dfa1[['transaction_amount', 'payment_method', 'account_age_days', 'customer_age']]
            y = dfa1['is_fraudulent']

            X = pd.get_dummies(X, columns=['payment_method'], drop_first=True)

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42
            )

            log_model = LogisticRegression(max_iter=100, random_state=42)
            log_model.fit(X_train, y_train)
            y_pred_log = log_model.predict(X_test)
            st.write("#### Logistic Regression Model Performance")
            st.text(classification_report(y_test, y_pred_log))

            rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            st.write("#### Random Forest Model Performance")
            st.text(classification_report(y_test, y_pred_rf))

            st.write("#### ROC Curve for Fraud Detection Models")
            def plot_roc_curve(model, X_test, y_test, label):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{label} AUC = {auc_score:.2f}")
            
            plt.figure(figsize=(10, 6))
            plot_roc_curve(log_model, X_test, y_test, "Logistic Regression")
            plot_roc_curve(rf_model, X_test, y_test, "Random Forest")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve for Fraud Detection Models")
            plt.legend(loc="best")
            st.pyplot(plt)

            feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
            plt.figure(figsize=(8, 6))
            feature_importances.sort_values().plot(kind="barh", color="teal")
            plt.title("Feature Importance in Predicting Fraudulent Transactions")
            plt.xlabel("Feature Importance Score")
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Error during modeling: {e}")
    else:
        st.warning("No data available for visualizations.")
