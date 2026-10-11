from datetime import datetime
from typing import Optional, Union, List, Dict, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .common import BasePredictionResponse


class AccountHistoryItem(BaseModel):
    """Historical transaction record strictly prior to assessment timestamp."""
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: Union[datetime, str] = Field(..., description="Timestamp of past transaction")
    direction: Optional[Literal["inbound", "outbound"]] = Field(
        default="outbound",
        description="Whether funds moved in or out"
    )
    channel: Optional[str] = Field(default="online", description="Channel utilized")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_str(cls, v: Union[datetime, str]) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class AnomalyPredictRequest(BaseModel):
    """Request schema for POST /predict/anomaly.
    
    Accepts account ID and immediate transaction information alongside historical transactions.
    """
    account_id: str = Field(..., alias="senderAccountId", description="Account identifier under behavioral analysis")
    current_amount: float = Field(..., gt=0, le=10_000_000.0, description="Amount bounded to $10M")
    assessment_timestamp: Union[datetime, str] = Field(
        ...,
        alias="timestamp",
        description="Current evaluation cutoff timestamp (no future leakage permitted)"
    )
    channel: Optional[str] = Field(default="online", description="Channel of current transaction")
    history: Optional[List[AccountHistoryItem]] = Field(
        default_factory=list,
        max_length=1000,
        description="List of prior transactions for this account (bounded to 1000)"
    )
    min_history_required: int = Field(
        default=3,
        description="Threshold of historical transactions to exit cold-start"
    )

    @field_validator("assessment_timestamp")
    @classmethod
    def validate_timestamp_str(cls, v: Union[datetime, str]) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class AnomalyPredictResponse(BasePredictionResponse):
    """Response schema for POST /predict/anomaly."""
    account_id: str
    raw_anomaly_score: Optional[float] = Field(
        default=None,
        description="Raw Isolation Forest decision function score (negative indicates outlier)"
    )
    is_cold_start: bool = Field(
        default=False,
        description="True if account has insufficient history for behavioral baseline"
    )
    cold_start_policy_applied: Optional[str] = Field(
        default=None,
        description="Details of cold-start heuristic or neutral fallback applied"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
