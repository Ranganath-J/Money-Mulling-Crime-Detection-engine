import time
from fastapi import APIRouter, HTTPException, status
from ..logging import logger
from ..services.model_registry import registry
from ..schemas.transaction import TransactionPredictRequest, TransactionPredictResponse
from ..schemas.anomaly import AnomalyPredictRequest, AnomalyPredictResponse
from ..schemas.gnn import GNNPredictRequest, GNNPredictResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/transaction", response_model=TransactionPredictResponse)
def predict_transaction(req: TransactionPredictRequest) -> TransactionPredictResponse:
    """Predict risk for an individual transaction using the transaction ML model."""
    start_time = time.perf_counter()
    model_info = registry.models.get("transaction_classifier", {})
    instance = model_info.get("instance")

    if model_info.get("status") != "available" or instance is None:
        latency = (time.perf_counter() - start_time) * 1000
        logger.warning(f"Transaction prediction requested for {req.transaction_id} but model is unavailable")
        return TransactionPredictResponse(
            model_name=model_info.get("name", "TransactionClassifier"),
            model_version=model_info.get("version", "1.0.0"),
            status="unavailable",
            score=None,  # NEVER 0 for unavailable!
            transaction_id=req.transaction_id,
            latency_ms=round(latency, 2),
            error="Transaction classifier model artifact is not loaded or unavailable",
            reasons=["Model artifact missing or uninitialized"],
            explanation=None,
            evidence=None,
        )

    # Active inference execution (Prompt 2A.2)
    try:
        prediction_result = instance.predict_one(req)
        prediction_result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return prediction_result
    except Exception as e:
        latency = (time.perf_counter() - start_time) * 1000
        logger.exception(f"Error during transaction inference: {e}")
        return TransactionPredictResponse(
            model_name=model_info.get("name", "TransactionClassifier"),
            model_version=model_info.get("version", "1.0.0"),
            status="unavailable",
            score=None,
            transaction_id=req.transaction_id,
            latency_ms=round(latency, 2),
            error=f"Inference exception: {str(e)}",
            reasons=["Internal model error during evaluation"],
        )


@router.post("/anomaly", response_model=AnomalyPredictResponse)
def predict_anomaly(req: AnomalyPredictRequest) -> AnomalyPredictResponse:
    """Detect behavioral anomaly for an account based on history prior to assessment timestamp."""
    start_time = time.perf_counter()
    model_info = registry.models.get("anomaly_detector", {})
    instance = model_info.get("instance")

    if model_info.get("status") != "available" or instance is None:
        latency = (time.perf_counter() - start_time) * 1000
        logger.warning(f"Anomaly prediction requested for account {req.account_id} but model is unavailable")
        return AnomalyPredictResponse(
            model_name=model_info.get("name", "BehaviouralAnomalyDetector"),
            model_version=model_info.get("version", "1.0.0"),
            status="unavailable",
            score=None,  # NEVER 0 for unavailable!
            account_id=req.account_id,
            latency_ms=round(latency, 2),
            error="Anomaly detector model artifact is not loaded or unavailable",
            reasons=["Model artifact missing or uninitialized"],
            is_cold_start=len(req.history or []) < req.min_history_required,
            cold_start_policy_applied=None,
        )

    # Active inference execution (Prompt 2A.2)
    try:
        prediction_result = instance.predict_one(req)
        prediction_result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return prediction_result
    except Exception as e:
        latency = (time.perf_counter() - start_time) * 1000
        logger.exception(f"Error during anomaly inference: {e}")
        return AnomalyPredictResponse(
            model_name=model_info.get("name", "BehaviouralAnomalyDetector"),
            model_version=model_info.get("version", "1.0.0"),
            status="unavailable",
            score=None,
            account_id=req.account_id,
            latency_ms=round(latency, 2),
            error=f"Inference exception: {str(e)}",
            reasons=["Internal model error during evaluation"],
        )


@router.post("/gnn", response_model=GNNPredictResponse)
def predict_gnn(req: GNNPredictRequest) -> GNNPredictResponse:
    """Predict money muling network structural risk using PyTorch Geometric GNN."""
    start_time = time.perf_counter()
    model_info = registry.models.get("gnn_detector", {})
    instance = model_info.get("instance")

    if model_info.get("status") != "available" or instance is None:
        latency = (time.perf_counter() - start_time) * 1000
        logger.warning("GNN prediction requested but model is unavailable")
        return GNNPredictResponse(
            model_name=model_info.get("name", "PyTorchGeometricGNN"),
            model_version=model_info.get("version", "1.0.0"),
            status="unavailable",
            score=None,  # NEVER 0 for unavailable!
            transaction_id=req.target_transaction_id,
            subgraph_nodes_count=len(req.nodes),
            subgraph_edges_count=len(req.edges),
            latency_ms=round(latency, 2),
            error="GNN model checkpoint is not loaded or unavailable",
            reasons=["GNN checkpoint missing or uninitialized"],
        )

    # Active inference execution (Prompt 2A.3)
    try:
        prediction_result = instance.predict_subgraph(req)
        prediction_result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return prediction_result
    except Exception as e:
        latency = (time.perf_counter() - start_time) * 1000
        logger.exception(f"Error during GNN inference: {e}")
        return GNNPredictResponse(
            model_name=model_info.get("name", "PyTorchGeometricGNN"),
            model_version=model_info.get("version", "1.0.0"),
            status="unavailable",
            score=None,
            transaction_id=req.target_transaction_id,
            subgraph_nodes_count=len(req.nodes),
            subgraph_edges_count=len(req.edges),
            latency_ms=round(latency, 2),
            error=f"Inference exception: {str(e)}",
            reasons=["Internal model error during evaluation"],
        )
