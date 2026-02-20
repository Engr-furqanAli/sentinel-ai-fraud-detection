from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.response import PredictionResponse
from backend.app.schemas.transaction import RetailTransactionRequest
import joblib
import os
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Model paths
# Get project root directory safely
import pathlib

# Get project root folder (sentinel-fraud-platform)
BASE_DIR = pathlib.Path(__file__).resolve().parents[5]

MODEL_PATH = BASE_DIR / "data" / "models" / "retail_fraud_model.pkl"
ENCODERS_PATH = BASE_DIR / "data" / "models" / "retail_label_encoders.pkl"

MODEL_PATH = str(MODEL_PATH)
ENCODERS_PATH = str(ENCODERS_PATH)

# Global variables
model = None
label_encoders = None

# Expected features (should match training)
EXPECTED_FEATURES = [
    'sender_account', 'receiver_account', 'amount', 'transaction_type',
    'merchant_category', 'location', 'device_used', 'time_since_last_transaction',
    'spending_deviation_score', 'velocity_score', 'geo_anomaly_score',
    'payment_channel', 'ip_address', 'device_hash'
]

# Categorical columns that need encoding
CATEGORICAL_COLUMNS = [
    'sender_account', 'receiver_account', 'transaction_type',
    'merchant_category', 'location', 'device_used',
    'payment_channel', 'ip_address', 'device_hash'
]


def load_model_and_encoders():
    """Load model and label encoders with error handling"""
    global model, label_encoders

    try:
        # Load model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")

        # Load label encoders if they exist
        if os.path.exists(ENCODERS_PATH):
            label_encoders = joblib.load(ENCODERS_PATH)
            logger.info(f"Label encoders loaded from {ENCODERS_PATH}")
        else:
            logger.warning("Label encoders not found. Using fallback encoding method.")
            label_encoders = None

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


# Load model at startup
try:
    load_model_and_encoders()
except Exception as e:
    logger.error(f"Failed to load model on startup: {str(e)}")
    # Don't raise here - allow app to start but endpoints will fail gracefully


def validate_input_data(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean input data"""
    cleaned_data = {}

    for feature in EXPECTED_FEATURES:
        if feature not in transaction:
            # Set default values for missing features
            if feature in [
                'amount',
                'time_since_last_transaction',
                'spending_deviation_score',
                'velocity_score',
                'geo_anomaly_score'
            ]:
                cleaned_data[feature] = 0.0
            else:
                cleaned_data[feature] = "unknown"
        else:
            cleaned_data[feature] = transaction[feature]

    return cleaned_data


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features using saved encoders or fallback method"""

    if label_encoders is not None:
        # Use saved label encoders
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns and col in label_encoders:
                try:
                    # Convert to string and handle unknown categories
                    df[col] = df[col].astype(str)
                    # Map to -1 for unknown categories
                    df[col] = df[col].map(
                        lambda x: label_encoders[col].transform([x])[0]
                        if x in label_encoders[col].classes_
                        else -1
                    )
                except Exception as e:
                    logger.warning(f"Error encoding {col}: {str(e)}. Using fallback.")
                    df[col] = pd.factorize(df[col])[0]
            elif col in df.columns:
                # Fallback: use pandas factorize
                df[col] = pd.factorize(df[col].astype(str))[0]
    else:
        # Fallback encoding method
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = pd.factorize(df[col].astype(str))[0]

    return df


def calculate_risk(probability: float) -> tuple:
    """
    Calculate risk score and level based on fraud probability

    Args:
        probability: Fraud probability (0 to 1)

    Returns:
        tuple: (risk_score, risk_level, decision)
    """
    risk_score = round(probability * 100, 2)

    if probability < 0.15:
        return risk_score, "Low", "Approve"
    elif probability < 0.35:
        return risk_score, "Medium", "Manual Review"
    else:
        return risk_score, "High", "Block Transaction"


@router.post(
    "/predict_retail",
    response_model=PredictionResponse,
    summary="Predict retail transaction fraud",
    description="Predicts fraud probability for a retail transaction and provides risk assessment",
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid input data"},
        500: {"description": "Model not loaded or prediction failed"}
    }
)
async def predict_retail(transaction: dict):
    """
    Predict fraud probability for a retail transaction

    - **transaction**: Dictionary containing transaction features
    - Returns: Fraud probability, risk score, risk level, and decision
    """

    # Check if model is loaded
    if model is None:
        try:
            load_model_and_encoders()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not available. Please try again later."
            )

    try:
        # Log incoming request (without sensitive data)
        logger.info(f"Received prediction request with keys: {list(transaction.keys())}")

        # Validate and clean input data
        cleaned_transaction = validate_input_data(transaction)

        # Convert to DataFrame
        df = pd.DataFrame([cleaned_transaction])

        # Ensure all expected features are present and in correct order
        df = df[EXPECTED_FEATURES]

        # Encode categorical features
        df = encode_categorical_features(df)

        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle any NaN values
        if df.isnull().any().any():
            logger.warning("NaN values detected after conversion. Filling with 0.")
            df = df.fillna(0)

        # Make prediction
        probability = model.predict_proba(df)[0][1]
        probability = round(float(probability), 4)

        # Calculate risk metrics
        risk_score, risk_level, decision = calculate_risk(probability)

        # Log prediction result
        logger.info(f"Prediction successful - Probability: {probability}, Decision: {decision}")

        return PredictionResponse(
            fraud_probability=probability,
            risk_score=risk_score,
            risk_level=risk_level,
            decision=decision
        )

    except ValueError as e:
        logger.error(f"ValueError in prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )
    except KeyError as e:
        logger.error(f"KeyError in prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required field: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during prediction"
        )


@router.get("/model_info", summary="Get model information")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return {
        "model_type": type(model).__name__,
        "features": EXPECTED_FEATURES,
        "categorical_features": CATEGORICAL_COLUMNS,
        "encoders_loaded": label_encoders is not None
    }


@router.post("/reload_model", summary="Reload the model")
async def reload_model():
    """Reload the model and encoders"""
    try:
        load_model_and_encoders()
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )