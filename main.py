
import pandas as pd

# Load dataset
data = pd.read_csv("dataset/creditcard.csv")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape (rows, columns):")
print(data.shape)

print("\nTransaction counts:")
print(data["Class"].value_counts())

fraud_percentage = (data["Class"].sum() / len(data)) * 100
print(f"\nFraud percentage: {fraud_percentage:.4f}%")

import matplotlib.pyplot as plt
import seaborn as sns

# Count genuine and fraudulent transactions
class_counts = data["Class"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
sns.barplot(
    x=["Genuine", "Fraud"],
    y=[class_counts[0], class_counts[1]]
)

plt.title("Genuine vs Fraudulent Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()

# Transaction amount distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=data,
    x="Amount",
    hue="Class",
    bins=50
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
# Data Analysis and Cleaning

print("\n--- Dataset Information ---")
print(data.info())

print("\n--- Missing Values ---")
print(data.isnull().sum().sum())

print("\n--- Duplicate Rows ---")
print(data.duplicated().sum())

print("\n--- Statistical Summary ---")
print(data[["Time", "Amount"]].describe())

# Step 2: Data Cleaning and Preprocessing

print("\n--- Data Cleaning ---")

# Check duplicate rows
duplicate_count = data.duplicated().sum()
print("Duplicate rows found:", duplicate_count)

# Remove duplicate rows
data = data.drop_duplicates().copy()

print("Dataset shape after removing duplicates:", data.shape)

# Check missing values
print("Total missing values:", data.isnull().sum().sum())

# Check genuine and fraud transactions
print("\nTransaction counts after cleaning:")
print(data["Class"].value_counts())

print("\nData cleaning completed successfully!")

# Step 2: Data Cleaning

print("\n--- Data Cleaning ---")

duplicate_count = data.duplicated().sum()
print("Duplicate rows found:", duplicate_count)

data = data.drop_duplicates().copy()

print("Dataset shape after cleaning:", data.shape)
print("Missing values:", data.isnull().sum().sum())

print("\nTransaction counts after cleaning:")
print(data["Class"].value_counts())

print("\nData cleaning completed successfully!")

# Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Genuine vs Fraud Transactions
plt.figure(figsize=(7, 5))

sns.countplot(data=data, x="Class")

plt.title("Genuine vs Fraud Transactions")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.show()


# 2. Transaction Amount Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=data,
    x="Amount",
    hue="Class",
    bins=50
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Graph 1: Genuine vs Fraud Transactions
plt.figure(figsize=(7, 5))

sns.countplot(x="Class", data=data)

plt.title("Genuine vs Fraud Transactions")
plt.xlabel("Transaction Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.savefig("transaction_counts.png")
plt.show()

# Graph 2: Transaction Amount Distribution
plt.figure(figsize=(8, 5))

sns.histplot(data=data, x="Amount", bins=50)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("amount_distribution.png")
plt.show()

print("Graphs created successfully!")

