# train_retail_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import numpy as np

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
DATA_PATH = "../../data/raw/retaill_transactions.csv"  # update if your CSV has another name
MODEL_SAVE_PATH = "../../data/models/retail_fraud_model.pkl"

df = pd.read_csv(DATA_PATH)
print("Columns in dataset:", df.columns.tolist())
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# -----------------------------
# 2️⃣ Check for missing values
# -----------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

# -----------------------------
# 3️⃣ Define target and features
# -----------------------------
TARGET_COL = "is_fraud"
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# -----------------------------
# 4️⃣ Identify column types
# -----------------------------
# Columns to drop completely (IDs, timestamps that shouldn't be used for training)
columns_to_drop = ['transaction_id', 'timestamp', 'fraud_type']  # fraud_type is the target's explanation, not a feature

# Drop these columns if they exist
for col in columns_to_drop:
    if col in X.columns:
        X = X.drop(columns=[col])
        print(f"Dropped column: {col}")

# Identify categorical columns (object dtype or those with string values)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns to encode: {categorical_cols}")

# Identify numerical columns
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical columns: {numerical_cols}")

# -----------------------------
# 5️⃣ Encode categorical columns
# -----------------------------
# Create a dictionary to store label encoders (optional, if you need to inverse transform later)
label_encoders = {}

for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        # Convert to string and handle any NaN values
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"Encoded column: {col} with {len(le.classes_)} unique values")

# -----------------------------
# 6️⃣ Handle any remaining issues
# -----------------------------
# Check for any remaining non-numeric columns
remaining_object_cols = X.select_dtypes(include=['object']).columns.tolist()
if remaining_object_cols:
    print(f"Warning: Still have object columns: {remaining_object_cols}")
    # Force convert any remaining object columns to numeric (errors become NaN)
    for col in remaining_object_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Fill any NaN values that might have been created
if X.isnull().any().any():
    print("Filling NaN values with column means...")
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].mean(), inplace=True)

# -----------------------------
# 7️⃣ Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Features used for training: {X.columns.tolist()}")

# -----------------------------
# 8️⃣ Train RandomForest model
# -----------------------------
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=10,  # Limit depth to prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2
)

print("\nTraining model...")
clf.fit(X_train, y_train)

# -----------------------------
# 9️⃣ Save model
# -----------------------------
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(clf, MODEL_SAVE_PATH)
print(f"✅ Model trained and saved at {MODEL_SAVE_PATH}")

# -----------------------------
# 🔟 Optional: evaluate
# -----------------------------
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print(f"\nModel accuracy on training set: {train_score:.4f}")
print(f"Model accuracy on test set: {test_score:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))