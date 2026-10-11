from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class EvaluationMetrics(BaseModel):
    model_name: str
    model_version: str
    status: Literal["available", "unavailable", "evaluated", "error"]
    sample_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    false_positive_rate: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    notes: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class EvaluationRequest(BaseModel):
    """Request for running or retrieving reproducible model evaluation."""
    split: Literal["val", "test", "synthetic_benchmark"] = "test"
    dataset_seed: int = 42
    models: Optional[List[str]] = Field(
        default=None,
        description="Optional subset of models: ['transaction', 'anomaly', 'gnn', 'combined']"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class EvaluationResponse(BaseModel):
    status: Literal["completed", "partial", "unavailable"]
    timestamp: str
    dataset_seed: int
    split: str
    feature_schema_version: str
    models_evaluated: Dict[str, EvaluationMetrics]
    comparison_summary: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
