import pytest
from app.services.model_registry import registry
from app.models.transaction_classifier import TransactionClassifierModel
from app.models.anomaly_detector import BehaviouralAnomalyDetectorModel
from scripts.train_prompt_2a2 import generate_synthetic_transactions


def test_transaction_normal_prediction(client):
    registry.load_models_if_present()
    payload = {
        "id": "TX_NORM_101",
        "senderAccountId": "ACC_0012",
        "receiverAccountId": "ACC_0045",
        "amount": 75.50,
        "timestamp": "2026-10-09T14:30:00Z",
        "channel": "online",
        "context": {
            "recent_velocity": 1.0,
            "historical_avg_amount": 80.0,
            "historical_std_amount": 20.0,
            "inflow_outflow_ratio": 1.05,
            "account_age_days": 180
        }
    }
    response = client.post("/predict/transaction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "available"
    assert data["transaction_id"] == "TX_NORM_101"
    assert data["score"] is not None
    assert 0.0 <= data["score"] <= 100.0
    assert data["risk_category"] in ("Monitor", "Review", "Investigation Alert")
    assert "explanation" in data
    assert "evidence" in data
    assert len(data["reasons"]) > 0


def test_transaction_suspicious_prediction(client):
    registry.load_models_if_present()
    # High-amount, high-velocity mule pattern
    payload = {
        "id": "TX_SUSP_909",
        "senderAccountId": "ACC_MULE_1",
        "receiverAccountId": "ACC_DEST_2",
        "amount": 9850.0,
        "timestamp": "2026-10-09T03:15:00Z",
        "channel": "wire",
        "context": {
            "recent_velocity": 14.0,
            "historical_avg_amount": 85.0,
            "historical_std_amount": 25.0,
            "inflow_outflow_ratio": 1.0,
            "account_age_days": 8
        }
    }
    response = client.post("/predict/transaction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "available"
    assert data["score"] is not None
    assert data["score"] >= 70.0  # High risk score
    assert data["risk_category"] in ("Review", "Investigation Alert")


def test_anomaly_cold_start_policy(client):
    registry.load_models_if_present()
    # 0 historical transactions -> cold start
    payload = {
        "senderAccountId": "ACC_NEWBIE_01",
        "current_amount": 500.0,
        "timestamp": "2026-10-09T12:00:00Z",
        "history": []
    }
    response = client.post("/predict/anomaly", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_cold_start"] is True
    assert data["status"] == "fallback"
    assert data["score"] == 25.0
    assert "cold_start_policy_applied" in data
    assert data["cold_start_policy_applied"] is not None


def test_anomaly_established_account_deviation(client):
    registry.load_models_if_present()
    # Established account history with average around $50, but current tx is $15,000
    payload = {
        "senderAccountId": "ACC_ESTABLISHED",
        "current_amount": 15000.0,
        "timestamp": "2026-10-09T18:00:00Z",
        "history": [
            {"amount": 45.0, "timestamp": "2026-10-07T10:00:00Z", "direction": "outbound"},
            {"amount": 55.0, "timestamp": "2026-10-08T11:00:00Z", "direction": "outbound"},
            {"amount": 40.0, "timestamp": "2026-10-09T09:00:00Z", "direction": "outbound"},
            {"amount": 60.0, "timestamp": "2026-10-09T12:00:00Z", "direction": "outbound"},
        ]
    }
    response = client.post("/predict/anomaly", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_cold_start"] is False
    assert data["status"] == "available"
    assert data["score"] is not None
    assert data["score"] > 50.0  # Anomaly detected
    assert data["raw_anomaly_score"] is not None


def test_missing_model_artifact_returns_unavailable(monkeypatch, client):
    # Simulate missing artifact
    original_model = registry.models["transaction_classifier"]["instance"]
    original_status = registry.models["transaction_classifier"]["status"]
    try:
        registry.models["transaction_classifier"]["status"] = "unavailable"
        registry.models["transaction_classifier"]["instance"] = None
        
        payload = {
            "id": "TX_TEST_UNAVAIL",
            "senderAccountId": "ACC_A",
            "receiverAccountId": "ACC_B",
            "amount": 100.0,
            "timestamp": "2026-10-09T12:00:00Z",
        }
        response = client.post("/predict/transaction", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["score"] is None  # CRITICAL: score must NOT be 0
        assert "error" in data
    finally:
        registry.models["transaction_classifier"]["instance"] = original_model
        registry.models["transaction_classifier"]["status"] = original_status


def test_deterministic_repeatability():
    # Two synthetic data generations with the same seed must produce identical data
    rec1, lab1 = generate_synthetic_transactions(n_samples=50, seed=123)
    rec2, lab2 = generate_synthetic_transactions(n_samples=50, seed=123)
    assert rec1 == rec2
    assert (lab1 == lab2).all()


def test_class_imbalance_handling():
    # Verify class weights are configured in model pipeline
    model = TransactionClassifierModel(model_type="logistic_regression", random_state=42)
    clf = model.pipeline.named_steps["classifier"]
    assert clf.class_weight == "balanced"
