from typing import Any, Dict
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # features must match the training table columns (X) used by the model
    features: Dict[str, Any] = Field(
        ...,
        description="Customer-level features used by the trained model",
        examples=[{"total_amount": 1000.0, "avg_amount": 250.0, "tx_count": 4}],
    )


class PredictResponse(BaseModel):
    risk_probability: float
    credit_score: int
    model_uri: str
