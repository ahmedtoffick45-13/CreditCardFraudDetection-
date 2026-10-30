
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_model.pkl")

# Load dataset
data = pd.read_csv("dataset/creditcard.csv")
data = data.drop_duplicates().copy()

# Select one transaction for testing
X = data.drop("Class", axis=1)
y = data["Class"]

# Select a transaction known to be fraudulent, if available
fraud_rows = data[data["Class"] == 1]

if len(fraud_rows) > 0:
    transaction = fraud_rows.drop(columns=["Class"]).iloc[[0]]
    actual_class = 1

    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0][1]

    print("Actual transaction: Fraud")
    print("Model prediction:",
          "Fraud" if prediction == 1 else "Genuine")
    print(f"Predicted fraud probability: {probability:.2%}")
else:
    print("No fraud transaction found in dataset.")
