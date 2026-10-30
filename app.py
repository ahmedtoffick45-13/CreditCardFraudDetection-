
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

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
dataset_path = os.path.join("dataset", "creditcard_sample.csv")
model_path = "fraud_model.pkl"

# ---------------- LOAD MODEL ----------------
model = None

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
else:
    st.error("Model file fraud_model.pkl was not found.")

# ---------------- LOAD DATASET ----------------
if not os.path.exists(dataset_path):
    st.error("Dataset not found. Please check dataset/creditcard_sample.csv")
    st.stop()

try:
    data = pd.read_csv(dataset_path)
except Exception as e:
    st.error(f"Could not read dataset: {e}")
    st.stop()

if "Class" not in data.columns:
    st.error("Dataset must contain a Class column.")
    st.stop()

if len(data) == 0:
    st.error("Dataset is empty.")
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

bars = ax.bar(
    chart_data["Transaction Type"],
    chart_data["Count"]
)

ax.set_title("Genuine vs Fraud Transactions")
ax.set_xlabel("Transaction Type")
ax.set_ylabel("Number of Transactions")

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
st.pyplot(fig)
plt.close(fig)

# ---------------- FRAUD PERCENTAGE ----------------
st.divider()
st.subheader("📈 Fraud Percentage")

st.write(
    f"Fraud transactions make up **{fraud_percent:.4f}%** "
    "of the sample dataset."
)

# ---------------- DATASET PREVIEW ----------------
st.divider()
st.subheader("📋 Dataset Preview")

st.dataframe(data.head(10), use_container_width=True)

# ---------------- MODEL INFORMATION ----------------
st.divider()
st.subheader("🤖 Machine Learning Model")

if model is not None:
    st.success("Machine Learning model loaded successfully!")
    st.write("Model: Logistic Regression")
    st.write("The model is ready to predict transactions.")

    st.caption(
        "The prediction model was trained separately using the original dataset. "
        "The dashboard displays a smaller sample dataset."
    )
else:
    st.warning("Model unavailable. Check fraud_model.pkl.")

# ---------------- FRAUD PREDICTION ----------------
st.divider()
st.subheader("🔍 Machine Learning Fraud Prediction")

st.write(
    "Select a transaction row to check whether it is genuine or fraudulent."
)

row_number = st.number_input(
    "Enter transaction row number",
    min_value=0,
    max_value=len(data) - 1,
    value=0,
    step=1
)

if st.button("Predict Transaction", type="primary"):

    if model is None:
        st.error("Model unavailable. Please check fraud_model.pkl.")

    else:
        transaction = data.iloc[[int(row_number)]]
        X = transaction.drop(columns=["Class"])

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

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            
