from fastapi import FastAPI

from backend.app.api.v1.endpoints import (
    predict,
    explain,
    transactions,
    health,
    predict_retail
)

app = FastAPI(
    title="Sentinel Fraud Detection Platform",
    version="1.0.0"
)

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Credit Fraud"])
app.include_router(predict_retail.router, prefix="/api/v1", tags=["Retail Fraud"])
app.include_router(explain.router, prefix="/api/v1", tags=["Explain"])
app.include_router(transactions.router, prefix="/api/v1", tags=["Transactions"])
