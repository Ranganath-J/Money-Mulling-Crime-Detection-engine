import time
from typing import Dict, Any, Optional
from pathlib import Path
from ..config import settings
from ..logging import logger
from ..schemas.common import ModelStatus
from ..schemas.transaction import TransactionPredictRequest, TransactionPredictResponse
from ..schemas.anomaly import AnomalyPredictRequest, AnomalyPredictResponse
from ..schemas.gnn import GNNPredictRequest, GNNPredictResponse


class ModelRegistry:
    """Manages model artifacts, lifecycles, and health statuses.
    
    Strictly follows contract rules:
    - Never fabricates fake predictions if a model is unavailable.
    - Never returns 0 to represent an unavailable score (returns None).
    - Exposes model_name, model_version, and explicit status.
    """

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {
            "transaction_classifier": {
                "name": "TransactionClassifier",
                "version": "1.0.0",
                "status": "unavailable",
                "instance": None,
                "artifact_path": str(settings.transaction_model_path),
                "last_evaluated": None,
            },
            "anomaly_detector": {
                "name": "BehaviouralAnomalyDetector",
                "version": "1.0.0",
                "status": "unavailable",
                "instance": None,
                "artifact_path": str(settings.anomaly_model_path),
                "last_evaluated": None,
            },
            "gnn_detector": {
                "name": "PyTorchGeometricGNN",
                "version": "1.0.0",
                "status": "unavailable",
                "instance": None,
                "artifact_path": str(settings.gnn_model_path),
                "last_evaluated": None,
            },
        }

    def load_models_if_present(self):
        """Attempt to load persisted model artifacts from disk if they exist."""
        # Check transaction classifier artifact
        if settings.transaction_model_path.exists():
            try:
                import joblib
                loaded = joblib.load(settings.transaction_model_path)
                self.models["transaction_classifier"]["instance"] = loaded
                self.models["transaction_classifier"]["status"] = "available"
                logger.info(f"Loaded transaction classifier artifact from {settings.transaction_model_path}")
            except Exception as e:
                logger.error(f"Failed to load transaction classifier artifact: {e}")
                self.models["transaction_classifier"]["status"] = "unavailable"
        else:
            self.models["transaction_classifier"]["status"] = "unavailable"

        # Check anomaly detector artifact
        if settings.anomaly_model_path.exists():
            try:
                import joblib
                loaded = joblib.load(settings.anomaly_model_path)
                self.models["anomaly_detector"]["instance"] = loaded
                self.models["anomaly_detector"]["status"] = "available"
                logger.info(f"Loaded anomaly detector artifact from {settings.anomaly_model_path}")
            except Exception as e:
                logger.error(f"Failed to load anomaly detector artifact: {e}")
                self.models["anomaly_detector"]["status"] = "unavailable"
        else:
            self.models["anomaly_detector"]["status"] = "unavailable"

        # Check GNN model artifact
        if settings.gnn_model_path.exists():
            try:
                from ..models.gnn_detector import GNNModelWrapper
                gnn_wrapper = GNNModelWrapper()
                if gnn_wrapper.load_checkpoint(settings.gnn_model_path):
                    self.models["gnn_detector"]["instance"] = gnn_wrapper
                    self.models["gnn_detector"]["status"] = "available"
                    logger.info(f"Loaded GNN checkpoint from {settings.gnn_model_path}")
                else:
                    self.models["gnn_detector"]["status"] = "unavailable"
            except Exception as e:
                logger.error(f"Failed to load GNN model artifact: {e}")
                self.models["gnn_detector"]["status"] = "unavailable"
        else:
            self.models["gnn_detector"]["status"] = "unavailable"

    def get_model_status(self, model_key: str) -> Dict[str, Any]:
        info = self.models.get(model_key, {})
        return {
            "name": info.get("name", "Unknown"),
            "version": info.get("version", "0.0.0"),
            "status": info.get("status", "unavailable"),
            "artifact_present": Path(info.get("artifact_path", "")).exists() if info.get("artifact_path") else False,
        }

    def get_overall_health(self) -> Dict[str, Any]:
        statuses = {k: self.get_model_status(k) for k in self.models}
        any_available = any(v["status"] == "available" for v in statuses.values())
        all_available = all(v["status"] == "available" for v in statuses.values())
        if all_available:
            overall_status = "ok"
        elif any_available:
            overall_status = "degraded"
        else:
            overall_status = "unavailable"

        return {
            "status": overall_status,
            "models": statuses
        }


# Singleton instance across application
registry = ModelRegistry()
