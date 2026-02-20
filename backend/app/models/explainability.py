import joblib
import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
EXPLAINER_PATH = os.path.join(BASE_DIR, "data", "models", "shap_explainer.pkl")


class Explainer:
    def __init__(self):
        if not os.path.exists(EXPLAINER_PATH):
            raise FileNotFoundError(f"SHAP explainer not found at {EXPLAINER_PATH}")
        self.explainer = joblib.load(EXPLAINER_PATH)

    def explain(self, features: dict):
        X = np.array(list(features.values())).reshape(1, -1)
        shap_values = self.explainer(X)
        return {f"V{i+1}": float(val) for i, val in enumerate(shap_values.values[0])}
