from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional


class PredictionResult(BaseModel):
    """Single spectrum prediction result."""
    model_config = ConfigDict(protected_namespaces=())
    filename: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    drift_score: float
    model_version: str

class PredictResponse(BaseModel):
    """Response for both single and bulk predictions."""
    model_config = ConfigDict(protected_namespaces=())
    predictions: List[PredictionResult]
    total: int
    processing_time_ms: float


class FeedbackRequest(BaseModel):
    """Ground truth feedback for a prediction."""
    filename: str
    ground_truth: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str


class ReadyResponse(BaseModel):
    """Readiness check response."""
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_version: Optional[str] = None
    model_name: Optional[str] = None
