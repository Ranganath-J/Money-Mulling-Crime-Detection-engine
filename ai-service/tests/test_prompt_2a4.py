import pytest
from app.services.model_registry import registry


def test_payload_bounds_transaction_max_amount(client):
    # Exceeding $10,000,000 should be rejected by input validation
    payload = {
        "id": "TX_TOO_LARGE",
        "senderAccountId": "ACC_A",
        "receiverAccountId": "ACC_B",
        "amount": 15_000_000.0,
        "timestamp": "2026-10-09T10:00:00Z"
    }
    response = client.post("/predict/transaction", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "invalid_payload"


def test_payload_bounds_gnn_empty_nodes(client):
    # Nodes array cannot be empty (min_length=1)
    payload = {
        "targetTransactionId": "TX_EMPTY",
        "nodes": [],
        "edges": []
    }
    response = client.post("/predict/gnn", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "invalid_payload"


def test_comprehensive_evaluation_pipeline(client):
    registry.load_models_if_present()
    payload = {
        "split": "test",
        "dataset_seed": 42
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("completed", "partial")
    assert "models_evaluated" in data
    
    models = data["models_evaluated"]
    assert "transaction_classifier_only" in models
    assert "anomaly_detector_only" in models
    assert "gnn_only" in models
    assert "combined_fusion_model" in models

    # Verify metrics fields are present and valid
    tx_m = models["transaction_classifier_only"]
    assert tx_m["status"] == "evaluated"
    assert tx_m["precision"] is not None
    assert tx_m["recall"] is not None
    assert tx_m["f1"] is not None
    assert tx_m["roc_auc"] is not None
    assert tx_m["pr_auc"] is not None
    assert tx_m["latency_p50_ms"] is not None

    fusion_m = models["combined_fusion_model"]
    assert fusion_m["f1"] is not None


def test_inference_exception_safety(monkeypatch, client):
    # Simulate internal inference error in model
    class BrokenModel:
        def predict_one(self, req):
            raise RuntimeError("Hardware tensor computation fault")

    original = registry.models["transaction_classifier"]["instance"]
    try:
        registry.models["transaction_classifier"]["instance"] = BrokenModel()
        payload = {
            "id": "TX_BROKEN_TEST",
            "senderAccountId": "ACC_A",
            "receiverAccountId": "ACC_B",
            "amount": 500.0,
            "timestamp": "2026-10-09T12:00:00Z"
        }
        response = client.post("/predict/transaction", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["score"] is None
        assert "Inference exception" in data["error"]
    finally:
        registry.models["transaction_classifier"]["instance"] = original
