from fastapi import APIRouter
from .health import router as health_router
from .predict import router as predict_router
from .evaluate import router as evaluate_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(predict_router)
api_router.include_router(evaluate_router)

__all__ = ["api_router"]
