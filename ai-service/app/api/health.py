from datetime import datetime, timezone
from fastapi import APIRouter
from ..config import settings
from ..schemas.common import HealthResponse
from ..services.model_registry import registry

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """System health check endpoint providing model availability and system telemetry."""
    registry.load_models_if_present()
    health_info = registry.get_overall_health()
    
    return HealthResponse(
        status=health_info["status"],
        app_name=settings.app_name,
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        models=health_info["models"],
        environment=settings.app_env,
    )
