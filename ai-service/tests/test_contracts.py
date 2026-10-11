def test_transaction_endpoint_unavailable_state(client):
    payload = {
        "id": "TX_TEST_001",
        "senderAccountId": "ACC_ALPHA",
        "receiverAccountId": "ACC_BETA",
        "amount": 450.0,
        "timestamp": "2026-10-09T15:00:00Z",
        "channel": "online"
    }
    response = client.post("/predict/transaction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX_TEST_001"
    assert data["model_name"] == "TransactionClassifier"
    assert data["model_version"] == "1.0.0"
    # CONTRACT RULE: status must be explicit, score must NEVER be 0 for unavailable
    assert data["status"] in ("available", "unavailable")
    if data["status"] == "unavailable":
        assert data["score"] is None
        assert "error" in data


def test_anomaly_endpoint_unavailable_state(client):
    payload = {
        "senderAccountId": "ACC_ANOM_001",
        "current_amount": 12000.0,
        "timestamp": "2026-10-09T16:00:00Z",
        "history": []
    }
    response = client.post("/predict/anomaly", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["account_id"] == "ACC_ANOM_001"
    assert data["model_name"] == "BehaviouralAnomalyDetector"
    assert data["status"] in ("available", "unavailable", "fallback")
    if data["status"] == "unavailable":
        assert data["score"] is None


def test_gnn_endpoint_unavailable_state(client):
    payload = {
        "targetTransactionId": "TX_GNN_001",
        "nodes": [{"id": "ACC_1"}, {"id": "ACC_2"}],
        "edges": [{"source": "ACC_1", "target": "ACC_2", "amount": 8000.0, "timestamp": "2026-10-09T16:00:00Z"}]
    }
    response = client.post("/predict/gnn", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "PyTorchGeometricGNN"
    assert data["status"] in ("available", "unavailable")
    if data["status"] == "unavailable":
        assert data["score"] is None


def test_evaluate_endpoint(client):
    payload = {
        "split": "test",
        "dataset_seed": 42
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_evaluated" in data
    assert "transaction_classifier" in data["models_evaluated"]


def test_invalid_payload_returns_422(client):
    # Missing required amount and accounts
    response = client.post("/predict/transaction", json={"foo": "bar"})
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "invalid_payload"
