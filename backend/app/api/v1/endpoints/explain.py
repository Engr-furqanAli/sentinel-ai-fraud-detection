from fastapi import APIRouter
from backend.app.schemas.transaction import Transaction
from backend.app.models.explainability import Explainer

router = APIRouter()
explainer = Explainer()

@router.post("/", summary="Explain fraud prediction")
def explain(transaction: Transaction):
    shap_result = explainer.explain(transaction.dict())
    return {
        "explanation": shap_result
    }
