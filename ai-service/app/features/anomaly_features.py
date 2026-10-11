from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np


ANOMALY_FEATURE_NAMES = [
    "current_amount",
    "hist_mean_amount",
    "amount_deviation_ratio",
    "recent_frequency_24h",
    "hours_since_last_tx",
    "outbound_ratio",
]


class AnomalyFeatureExtractor:
    """Extracts behavioral baselines strictly from events prior to assessment timestamp.
    
    Prevents temporal data leakage and handles cold-start accounts gracefully.
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
    def extract_features(
        cls,
        current_amount: float,
        assessment_timestamp: Any,
        history: List[Any],
        min_history_required: int = 3,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Compute behavioral features from prior history only."""
        curr_dt = cls.parse_datetime(assessment_timestamp)
        
        # Filter history to strictly strictly strictly prior events
        valid_history = []
        for item in history or []:
            if hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            elif isinstance(item, dict):
                item_dict = item
            else:
                continue

            item_dt = cls.parse_datetime(item_dict.get("timestamp"))
            if item_dt <= curr_dt:
                valid_history.append((item_dt, float(item_dict.get("amount", 0.0)), item_dict.get("direction", "outbound")))

        is_cold_start = len(valid_history) < min_history_required

        if is_cold_start:
            # Cold start baseline: fallback defaults
            feats = np.array([
                current_amount,
                current_amount,
                0.0,
                1.0,
                24.0,
                1.0,
            ], dtype=np.float32).reshape(1, -1)
            summary = {
                "history_count": len(valid_history),
                "is_cold_start": True,
                "note": "Insufficient historical transactions for reliable behavioral baseline."
            }
            return feats, True, summary

        # Sort history by timestamp descending
        valid_history.sort(key=lambda x: x[0], reverse=True)
        amounts = [h[1] for h in valid_history]
        hist_mean = float(np.mean(amounts))
        hist_std = float(np.std(amounts)) if len(amounts) > 1 else 1.0

        # Amount deviation ratio: abs(curr - mean) / (mean + 1e-4)
        amount_dev = float(abs(current_amount - hist_mean) / (hist_mean + 1e-4))

        # Recent frequency within rolling 24 hours
        freq_24h = sum(1 for h in valid_history if (curr_dt - h[0]).total_seconds() <= 86400)

        # Time since most recent transaction
        most_recent_dt = valid_history[0][0]
        hours_since = max(0.01, float((curr_dt - most_recent_dt).total_seconds() / 3600.0))

        # Outbound ratio
        outbound_count = sum(1 for h in valid_history if h[2] == "outbound")
        outbound_ratio = float(outbound_count / len(valid_history))

        feats = np.array([
            current_amount,
            hist_mean,
            amount_dev,
            float(freq_24h),
            hours_since,
            outbound_ratio,
        ], dtype=np.float32).reshape(1, -1)

        summary = {
            "history_count": len(valid_history),
            "historical_mean_amount": round(hist_mean, 2),
            "historical_std_amount": round(hist_std, 2),
            "amount_deviation_ratio": round(amount_dev, 2),
            "recent_24h_frequency": freq_24h,
            "hours_since_last_tx": round(hours_since, 2),
            "outbound_ratio": round(outbound_ratio, 2),
            "is_cold_start": False,
        }

        return feats, False, summary
