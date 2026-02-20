# frontend/dashboard.py
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Multi-Purpose Fraud Detection", layout="wide")
st.title("💳 Multi-Purpose Fraud Detection & Risk Scoring Dashboard")

# Sidebar - choose model
st.sidebar.header("Choose Dataset / Model")
model_option = st.sidebar.selectbox(
    "Select Dataset / Model",
    ["Credit Card Fraud", "Retail Transactions", "Custom Upload & Train"]
)
if model_option == "Credit Card Fraud":
    st.subheader("Enter Credit Card Transaction Features")

    input_data = {
        "Time": st.number_input("Time"),
        "V1": st.number_input("V1"),
        "V2": st.number_input("V2"),
        "V3": st.number_input("V3"),
        "V4": st.number_input("V4"),
        "V5": st.number_input("V5"),
        "V6": st.number_input("V6"),
        "V7": st.number_input("V7"),
        "V8": st.number_input("V8"),
        "V9": st.number_input("V9"),
        "V10": st.number_input("V10"),
        "V11": st.number_input("V11"),
        "V12": st.number_input("V12"),
        "V13": st.number_input("V13"),
        "V14": st.number_input("V14"),
        "V15": st.number_input("V15"),
        "V16": st.number_input("V16"),
        "V17": st.number_input("V17"),
        "V18": st.number_input("V18"),
        "V19": st.number_input("V19"),
        "V20": st.number_input("V20"),
        "V21": st.number_input("V21"),
        "V22": st.number_input("V22"),
        "V23": st.number_input("V23"),
        "V24": st.number_input("V24"),
        "V25": st.number_input("V25"),
        "V26": st.number_input("V26"),
        "V27": st.number_input("V27"),
        "V28": st.number_input("V28"),
        "Amount": st.number_input("Amount"),
    }

    if st.button("Predict"):
        API_URL = "http://127.0.0.1:8000/api/v1/predict"
        response = requests.post(API_URL, json=input_data).json()
        st.subheader("Prediction Results")
        st.json(response)
elif model_option == "Retail Transactions":
    st.subheader("Enter Retail Transaction Data")

    input_data = {
        "UserId": st.text_input("UserId"),
        "TransactionId": st.text_input("TransactionId"),
        "TransactionTime": st.text_input("TransactionTime"),
        "ItemCode": st.text_input("ItemCode"),
        "ItemDescription": st.text_input("ItemDescription"),
        "NumberOfItemsPurchased": st.number_input("Number of Items"),
        "CostPerItem": st.number_input("Cost Per Item"),
        "Country": st.text_input("Country"),
    }

    if st.button("Predict"):
        API_URL = "http://127.0.0.1:8000/api/v1/predict_retail"  # We’ll create this endpoint
        response = requests.post(API_URL, json=input_data).json()
        st.subheader("Prediction Results")
        st.json(response)
elif model_option == "Custom Upload & Train":
    st.subheader("Upload Your Own CSV for Training & Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview", df.head())

        target_col = st.text_input("Enter Target Column Name (e.g., Class)")

        if st.button("Train & Predict"):
            st.write("Training model on uploaded dataset...")

            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier

            X = df.drop(columns=[target_col])
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

            st.success("Model trained successfully!")

            pred = clf.predict(X_test.iloc[:1])
            prob = clf.predict_proba(X_test.iloc[:1])[:, 1]
            st.write("Example Prediction", pred[0])
            st.write("Predicted Probability", prob[0])
elif model_option == "Custom Upload & Train":
    st.subheader("Upload Your Own CSV for Training & Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview", df.head())

        target_col = st.text_input("Enter Target Column Name (e.g., Class)")

        if st.button("Train & Predict"):
            st.write("Training model on uploaded dataset...")

            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier

            X = df.drop(columns=[target_col])
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

            st.success("Model trained successfully!")

            pred = clf.predict(X_test.iloc[:1])
            prob = clf.predict_proba(X_test.iloc[:1])[:, 1]
            st.write("Example Prediction", pred[0])
            st.write("Predicted Probability", prob[0])
