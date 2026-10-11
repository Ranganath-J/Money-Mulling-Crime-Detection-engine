import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .config import settings
from .logging import logger
from .api import api_router
from .services.model_registry import registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: attempt to load existing artifacts
    logger.info("Initializing Money Mulling AI Service...")
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    registry.load_models_if_present()
    yield
    # Shutdown
    logger.info("Shutting down Money Mulling AI Service...")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="High-performance, interpretable ML & GNN engine for detecting financial crime and money muling networks.",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_and_logging_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} ({process_time_ms:.2f}ms)"
        )
        return response
    except Exception as exc:
        process_time_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal AI Service Error",
                "detail": str(exc) if settings.debug else "An unexpected error occurred during processing.",
                "duration_ms": round(process_time_ms, 2)
            }
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "status": "invalid_payload"
        }
    )


# Include API routes
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=settings.debug)
