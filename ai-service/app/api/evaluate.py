from datetime import datetime, timezone
from fastapi import APIRouter
from ..logging import logger
from ..services.model_registry import registry
from ..schemas.evaluation import EvaluationRequest, EvaluationResponse, EvaluationMetrics
from ..evaluation.pipeline import run_comprehensive_evaluation

router = APIRouter(tags=["Evaluation"])


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate_models(req: EvaluationRequest) -> EvaluationResponse:
    """Run reproducible offline evaluation comparison across models."""
    try:
        report = run_comprehensive_evaluation(seed=req.dataset_seed, split=req.split)
        metrics_dict = {
            k: EvaluationMetrics(**v) for k, v in report["models"].items()
        }
        available_count = sum(1 for m in metrics_dict.values() if m.status in ("available", "evaluated"))
        overall_status = "completed" if available_count == len(metrics_dict) else ("partial" if available_count > 0 else "unavailable")

        return EvaluationResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_seed=req.dataset_seed,
            split=req.split,
            feature_schema_version="1.0.0",
            models_evaluated=metrics_dict,
            comparison_summary={
                "models_evaluated_count": available_count,
                "total_models_registered": len(metrics_dict),
                "evaluation_configuration": report.get("evaluation_configuration", {}),
            }
        )
    except Exception as e:
        logger.exception(f"Error during offline evaluation pipeline: {e}")
        return EvaluationResponse(
            status="unavailable",
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_seed=req.dataset_seed,
            split=req.split,
            feature_schema_version="1.0.0",
            models_evaluated={},
            comparison_summary={"error": str(e)}
        )
