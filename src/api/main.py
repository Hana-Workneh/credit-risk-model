import os
from typing import Any, Dict, List

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictRequest, PredictResponse

# Configurable via env vars (nice for Docker / CI)
MODEL_URI = os.getenv("MODEL_URI", "models:/credit-risk-model@Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

app = FastAPI(title="Credit Risk API", version="1.0.0")

_model = None
_expected_cols: List[str] = []


def probability_to_score(prob: float, min_score: int = 300, max_score: int = 850) -> int:
    """Higher score = lower risk."""
    p = float(np.clip(prob, 0.0, 1.0))
    score = max_score - int(round(p * (max_score - min_score)))
    return int(np.clip(score, min_score, max_score))


def _infer_expected_columns(model) -> List[str]:
    """
    Best-effort: infer expected input columns from MLflow model signature.
    If signature is missing, we'll accept any columns and let model fail loudly.
    """
    try:
        sig = model.metadata.signature
        if sig and sig.inputs:
            return [col.name for col in sig.inputs.inputs]
    except Exception:
        pass
    return []


@app.on_event("startup")
def startup() -> None:
    global _model, _expected_cols
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _model = mlflow.pyfunc.load_model(MODEL_URI)
    _expected_cols = _infer_expected_columns(_model)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_uri": MODEL_URI, "expected_columns": _expected_cols}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert dict -> 1-row DataFrame
    X = pd.DataFrame([req.features])

    # Validate columns if we have a signature
    if _expected_cols:
        missing = [c for c in _expected_cols if c not in X.columns]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required feature columns: {missing}",
            )
        X = X[_expected_cols]

    try:
        pred = _model.predict(X)
        proba = float(np.asarray(pred).reshape(-1)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")

    return PredictResponse(
        risk_probability=proba,
        credit_score=probability_to_score(proba),
        model_uri=MODEL_URI,
    )
