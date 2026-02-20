import joblib
import pandas as pd

model = joblib.load("../../data/models/xgb_fraud_model.pkl")

df = pd.read_csv("../../data/raw/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

print("Loaded model successfully.")
