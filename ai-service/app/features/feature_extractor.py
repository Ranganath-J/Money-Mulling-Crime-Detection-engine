from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


CHANNEL_MAPPING = {
    "online": 0,
    "mobile": 1,
    "atm": 2,
    "wire": 3,
    "pos": 4,
    "branch": 5,
}

FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "channel_code",
    "recent_velocity",
    "amount_zscore",
    "inflow_outflow_ratio",
    "account_age_days",
]


class TransactionFeatureExtractor:
    """Extracts inference-time features from transaction payload and context.
    
    CRITICAL: Strictly excludes ground-truth labels and post-event information.
    """

    @staticmethod
    def parse_datetime(dt_val: Any) -> datetime:
        if isinstance(dt_val, datetime):
            return dt_val
        if isinstance(dt_val, str):
            try:
                return datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
            except Exception:
                pass
        return datetime.utcnow()

    @classmethod
    def extract_dict(cls, data: Dict[str, Any]) -> Dict[str, float]:
        amount = float(data.get("amount", 0.0))
        log_amount = float(np.log1p(max(0.0, amount)))
        
        dt = cls.parse_datetime(data.get("timestamp"))
        hour_of_day = float(dt.hour)
        day_of_week = float(dt.weekday())
        is_weekend = 1.0 if day_of_week >= 5 else 0.0

        channel_str = str(data.get("channel", "online")).lower()
        channel_code = float(CHANNEL_MAPPING.get(channel_str, 0))

        context = data.get("context") or {}
        if hasattr(context, "model_dump"):
            context = context.model_dump()

        recent_velocity = float(context.get("recent_velocity", 0.0))
        hist_avg = float(context.get("historical_avg_amount", amount))
        hist_std = float(context.get("historical_std_amount", 1.0))
        if hist_std <= 1e-4:
            hist_std = 1.0
        amount_zscore = float((amount - hist_avg) / hist_std)

        inflow_outflow_ratio = float(context.get("inflow_outflow_ratio", 1.0))
        account_age_days = float(context.get("account_age_days", 30))

        return {
            "amount": amount,
            "log_amount": log_amount,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "channel_code": channel_code,
            "recent_velocity": recent_velocity,
            "amount_zscore": amount_zscore,
            "inflow_outflow_ratio": inflow_outflow_ratio,
            "account_age_days": account_age_days,
        }

    @classmethod
    def extract_dataframe(cls, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        extracted = [cls.extract_dict(r) for r in rows]
        df = pd.DataFrame(extracted)
        return df[FEATURE_COLUMNS]

    @classmethod
    def extract_single_vector(cls, data: Dict[str, Any]) -> np.ndarray:
        feats = cls.extract_dict(data)
        return np.array([[feats[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
