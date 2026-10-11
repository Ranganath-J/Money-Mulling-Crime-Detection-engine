import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Money-Mulling-AI-Service"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8001
    debug: bool = False
    log_level: str = "INFO"

    # Timeouts & payload constraints
    request_timeout_seconds: float = 5.0
    max_batch_size: int = 1000
    max_graph_nodes: int = 5000
    max_graph_edges: int = 20000

    # Model Artifact Directories & Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    model_dir: Path = Path(__file__).resolve().parent / "models" / "artifacts"
    transaction_model_path: Path = Path(__file__).resolve().parent / "models" / "artifacts" / "transaction_classifier.joblib"
    anomaly_model_path: Path = Path(__file__).resolve().parent / "models" / "artifacts" / "anomaly_detector.joblib"
    gnn_model_path: Path = Path(__file__).resolve().parent / "models" / "artifacts" / "gnn_detector.pt"

    allow_fallback: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
