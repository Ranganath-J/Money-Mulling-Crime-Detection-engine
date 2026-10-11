from datetime import datetime
from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .common import BasePredictionResponse


class GraphNodePayload(BaseModel):
    id: str
    label: Optional[str] = None
    account_type: Optional[str] = "individual"
    inflow_sum: Optional[float] = 0.0
    outflow_sum: Optional[float] = 0.0
    degree: Optional[int] = 0
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class GraphEdgePayload(BaseModel):
    id: Optional[str] = None
    source: str
    target: str
    transaction_id: Optional[str] = Field(default=None, alias="transactionId")
    amount: float = Field(..., gt=0)
    timestamp: Union[datetime, str]
    channel: Optional[str] = "online"
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_str(cls, v: Union[datetime, str]) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class GNNPredictRequest(BaseModel):
    """Request schema for POST /predict/gnn.
    
    Accepts full graph or ego-subgraph of accounts (nodes) and transfers (directed edges).
    """
    target_transaction_id: Optional[str] = Field(
        default=None,
        alias="targetTransactionId",
        description="The specific transaction edge to evaluate"
    )
    nodes: List[GraphNodePayload] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Account nodes present in the subgraph (bounded to 5000)"
    )
    edges: List[GraphEdgePayload] = Field(
        ...,
        max_length=20000,
        description="Directed transfer edges between accounts (bounded to 20000)"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class EdgeRiskScore(BaseModel):
    transaction_id: Optional[str] = None
    edge_id: Optional[str] = None
    source: str
    target: str
    amount: float
    score: float = Field(..., ge=0.0, le=100.0)
    pattern: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class GNNPredictResponse(BasePredictionResponse):
    """Response schema for POST /predict/gnn."""
    transaction_id: Optional[str] = None
    subgraph_nodes_count: int = 0
    subgraph_edges_count: int = 0
    edge_scores: Optional[List[EdgeRiskScore]] = Field(
        default_factory=list,
        description="Risk scores computed for individual transaction edges in the graph"
    )
    detected_topologies: Optional[List[str]] = Field(
        default_factory=list,
        description="Graph patterns identified (e.g. 'cycle', 'fan-in', 'fan-out', 'pass-through')"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
