import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.model_registry import registry


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_registry():
    # Ensure fresh state
    registry.load_models_if_present()
    yield
