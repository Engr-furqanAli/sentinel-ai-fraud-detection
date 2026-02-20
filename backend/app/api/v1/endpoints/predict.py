from fastapi import APIRouter
from backend.app.schemas.transaction import Transaction
from backend.app.schemas.response import PredictionResponse
from backend.app.models.fraud_detector import FraudDetector
from backend.app.core.risk_engine import calculate_risk

router = APIRouter()
model = FraudDetector()


@router.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    # 1️⃣ Get fraud probability
    probability = model.predict(transaction.dict())

    # 2️⃣ Calculate risk using helper function
    risk_score, risk_level, decision = calculate_risk(probability)

    # 3️⃣ Return response
    return PredictionResponse(
        fraud_probability=probability,
        risk_score=risk_score,
        risk_level=risk_level,
        decision=decision
    )
