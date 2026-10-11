from datetime import datetime
from typing import Optional, Union, Dict, Any, List
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .common import BasePredictionResponse, RiskCategory


class TransactionContext(BaseModel):
    """Optional contextual and behavioural statistics provided by caller/feature-store."""
    recent_velocity: Optional[float] = Field(
        default=0.0,
        description="Transaction count for sender within rolling 1-hour window"
    )
    historical_avg_amount: Optional[float] = Field(
        default=0.0,
        description="Historical mean transaction amount for sender"
    )
    historical_std_amount: Optional[float] = Field(
        default=0.0,
        description="Historical standard deviation of transaction amount"
    )
    inflow_outflow_ratio: Optional[float] = Field(
        default=1.0,
        description="Ratio of funds in to funds out in previous 24h"
    )
    account_age_days: Optional[int] = Field(
        default=30,
        description="Sender account lifetime in days"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class TransactionPredictRequest(BaseModel):
    """Request schema for POST /predict/transaction.
    
    Accepts standard backend snake_case and frontend camelCase field names.
    Labels are STRICTLY prohibited from inference features.
    """
    transaction_id: str = Field(..., alias="id", description="Unique transaction ID")
    sender_id: str = Field(..., alias="senderAccountId", description="Sender account identifier")
    receiver_id: str = Field(..., alias="receiverAccountId", description="Receiver account identifier")
    amount: float = Field(..., gt=0, le=10_000_000.0, description="Transaction transfer amount bounded to $10M")
    currency: Optional[str] = Field(default="USD", description="Currency symbol/code")
    timestamp: Union[datetime, str] = Field(..., description="Timestamp of transaction")
    channel: Optional[str] = Field(default="online", description="Channel: online, mobile, atm, wire, pos")
    merchant_id: Optional[str] = Field(default=None, description="Optional merchant identifier")
    device_id: Optional[str] = Field(default=None, description="Optional device footprint identifier")
    context: Optional[TransactionContext] = Field(default_factory=TransactionContext)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_str(cls, v: Union[datetime, str]) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class TransactionPredictResponse(BasePredictionResponse):
    """Response schema for POST /predict/transaction."""
    transaction_id: str
    risk_category: Optional[RiskCategory] = Field(
        default=None,
        description="'Monitor', 'Review', or 'Investigation Alert'"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Model raw probability [0, 1] if calibrated"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
