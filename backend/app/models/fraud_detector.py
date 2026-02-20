import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../data/models/xgb_fraud_model.pkl"
)

class FraudDetector:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, features: dict) -> float:
        X = np.array(list(features.values())).reshape(1, -1)
        prob = self.model.predict_proba(X)[0][1]
        return float(prob)
