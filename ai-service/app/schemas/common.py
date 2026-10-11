from typing import Optional, Any, Dict, List, Literal
from pydantic import BaseModel, ConfigDict, Field

ModelStatus = Literal["available", "unavailable", "fallback", "degraded"]
RiskCategory = Literal["Monitor", "Review", "Investigation Alert"]


class BasePredictionResponse(BaseModel):
    """Base schema for all AI model prediction responses.
    
    Adheres strictly to the contract:
    - score is numeric 0-100 ONLY when a valid score exists.
    - score MUST be None (never 0) when unavailable.
    - status explicitly reflects model readiness.
    """
    model_name: str
    model_version: str
    status: ModelStatus
    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Confidence or risk presentation score scaled from 0 to 100. None if unavailable."
    )
    explanation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Top contributing features, SHAP approximations, or rule signals"
    )
    evidence: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structural or contextual evidence for downstream fusion"
    )
    reasons: Optional[List[str]] = Field(
        default_factory=list,
        description="Human-readable decision factors"
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Inference execution duration in milliseconds"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error detail when model inference is unavailable or degraded"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    app_name: str
    version: str
    timestamp: str
    models: Dict[str, Dict[str, Any]]
    environment: str

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
