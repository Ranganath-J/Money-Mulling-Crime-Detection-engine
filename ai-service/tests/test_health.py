def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ("ok", "degraded", "unavailable")
    assert data["app_name"] == "Money-Mulling-AI-Service"
    assert data["version"] == "1.0.0"
    assert "models" in data
    assert "transaction_classifier" in data["models"]
    assert "anomaly_detector" in data["models"]
    assert "gnn_detector" in data["models"]
