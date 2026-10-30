
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- DASHBOARD HEADING ----------------
st.title("🛡️ Credit Card Fraud Detection System")
st.write("Machine Learning Based Transaction Analysis Dashboard")
st.divider()

# ---------------- FILE PATHS ----------------
dataset_path = os.path.join("dataset", "creditcard.csv")
model_path = "fraud_model.pkl"

# ---------------- LOAD MODEL ----------------
model = None

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
else:
    st.warning("Model file fraud_model.pkl was not found.")

# ---------------- LOAD DATASET ----------------
if not os.path.exists(dataset_path):
    st.error("Dataset not found inside the dataset folder.")
    st.stop()

data = pd.read_csv(dataset_path)

if "Class" not in data.columns:
    st.error("Dataset must contain a Class column.")
    st.stop()

# ---------------- TRANSACTION STATISTICS ----------------
total = len(data)
fraud = int(data["Class"].sum())
genuine = total - fraud
fraud_percent = (fraud / total * 100) if total else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", f"{total:,}")
col2.metric("Genuine Transactions", f"{genuine:,}")
col3.metric("Fraud Transactions", f"{fraud:,}")

st.divider()

# ---------------- TRANSACTION DISTRIBUTION ----------------
st.subheader("📊 Transaction Distribution")

chart_data = pd.DataFrame({
    "Transaction Type": ["Genuine", "Fraud"],
    "Count": [genuine, fraud]
})

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(chart_data["Transaction Type"], chart_data["Count"])

ax.set_title("Genuine vs Fraud Transactions")
ax.set_xlabel("Transaction Type")
ax.set_ylabel("Number of Transactions")
ax.margins(y=0.12)

for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f"{int(height):,}",
        (bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center"
    )

fig.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

# ---------------- FRAUD PERCENTAGE ----------------
st.divider()
st.subheader("📈 Fraud Percentage")
st.write(
    f"Fraud transactions make up **{fraud_percent:.4f}%** "
    "of the dataset."
)

# ---------------- DATASET PREVIEW ----------------
st.subheader("📋 Dataset Preview")
st.dataframe(data.head(10), width="stretch")

# ---------------- MODEL PERFORMANCE ----------------
st.divider()
st.subheader("📊 Model Performance")

if model is not None:
    try:
        with st.spinner("Calculating model performance..."):

            # Match the split used in train_model.py
            clean_data = data.drop_duplicates().copy()
            X_all = clean_data.drop(columns=["Class"])
            y_all = clean_data["Class"]

            _, X_test, _, y_test = train_test_split(
                X_all,
                y_all,
                test_size=0.2,
                random_state=42,
                stratify=y_all
            )

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test, y_pred, zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, zero_division=0
            )
            f1 = f1_score(
                y_test, y_pred, zero_division=0
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{accuracy:.2%}")
            c2.metric("Precision", f"{precision:.2%}")
            c3.metric("Recall", f"{recall:.2%}")
            c4.metric("F1-Score", f"{f1:.2%}")

            st.write("**Confusion Matrix**")

            cm = confusion_matrix(
                y_test, y_pred, labels=[0, 1]
            )

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Genuine", "Actual Fraud"],
                columns=["Predicted Genuine", "Predicted Fraud"]
            )

            st.dataframe(cm_df, width="stretch")

            st.caption(
                "Accuracy alone can be misleading because fraud cases "
                "are rare. Review precision, recall and F1-score too."
            )

    except Exception as e:
        st.error(f"Could not calculate model performance: {e}")

else:
    st.warning("Model performance unavailable. Check fraud_model.pkl.")

# ---------------- MACHINE LEARNING PREDICTION ----------------
st.divider()
st.subheader("🤖 Machine Learning Fraud Prediction")

st.write(
    "Select a transaction row to predict whether it is genuine or fraudulent."
)

row_number = st.number_input(
    "Enter transaction row number",
    min_value=0,
    max_value=len(data) - 1,
    value=0,
    step=1,
    key="prediction_row"
)

if st.button("Predict Transaction", type="primary"):

    if model is None:
        st.error("Model unavailable. Check fraud_model.pkl.")

    else:
        transaction = data.iloc[[int(row_number)]]
        X = transaction.drop(columns=["Class"], errors="ignore")

        try:
            prediction = model.predict(X)[0]

            if prediction == 1:
                st.error("🚨 Model Prediction: FRAUDULENT TRANSACTION")
            else:
                st.success("✅ Model Prediction: GENUINE TRANSACTION")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    fraud_index = classes.index(1)
                    st.metric(
                        "Predicted Fraud Probability",
                        f"{probabilities[fraud_index]:.2%}"
                    )

            actual_label = int(data.iloc[int(row_number)]["Class"])

            st.write(
                "**Actual Dataset Label:**",
                "FRAUD" if actual_label == 1 else "GENUINE"
            )

            st.caption(
                "Prediction is generated by the trained model. "
                "The actual dataset label is shown for comparison."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

            

