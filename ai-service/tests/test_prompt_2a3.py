import pytest
from app.services.model_registry import registry
from app.models.gnn_detector import GNNModelWrapper


def test_gnn_normal_graph_inference(client):
    registry.load_models_if_present()
    payload = {
        "targetTransactionId": "TX_GNN_NORM_1",
        "nodes": [
            {"id": "ACC_A", "label": "Account A", "inflow_sum": 500.0, "outflow_sum": 450.0, "degree": 2},
            {"id": "ACC_B", "label": "Account B", "inflow_sum": 300.0, "outflow_sum": 200.0, "degree": 1},
            {"id": "ACC_C", "label": "Account C", "inflow_sum": 200.0, "outflow_sum": 350.0, "degree": 1},
        ],
        "edges": [
            {"id": "E_1", "transactionId": "TX_GNN_NORM_1", "source": "ACC_A", "target": "ACC_B", "amount": 150.0, "timestamp": "2026-10-09T10:00:00Z"},
            {"id": "E_2", "transactionId": "TX_GNN_NORM_2", "source": "ACC_B", "target": "ACC_C", "amount": 80.0, "timestamp": "2026-10-09T11:00:00Z"},
        ]
    }
    response = client.post("/predict/gnn", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "available"
    assert data["model_name"] == "PyTorchGeometricGNN"
    assert data["transaction_id"] == "TX_GNN_NORM_1"
    assert data["score"] is not None
    assert 0.0 <= data["score"] <= 100.0
    assert len(data["edge_scores"]) == 2
    # Check transaction_id mapping is strictly preserved
    assert data["edge_scores"][0]["transaction_id"] == "TX_GNN_NORM_1"


def test_gnn_cycle_topology_detection(client):
    registry.load_models_if_present()
    # Cycle: A -> B -> C -> A
    payload = {
        "targetTransactionId": "TX_CYC_1",
        "nodes": [
            {"id": "ACC_CYC_A", "inflow_sum": 8000.0, "outflow_sum": 8000.0, "degree": 2},
            {"id": "ACC_CYC_B", "inflow_sum": 8000.0, "outflow_sum": 8000.0, "degree": 2},
            {"id": "ACC_CYC_C", "inflow_sum": 8000.0, "outflow_sum": 8000.0, "degree": 2},
        ],
        "edges": [
            {"id": "E_C1", "transactionId": "TX_CYC_1", "source": "ACC_CYC_A", "target": "ACC_CYC_B", "amount": 8000.0, "timestamp": "2026-10-09T02:00:00Z"},
            {"id": "E_C2", "transactionId": "TX_CYC_2", "source": "ACC_CYC_B", "target": "ACC_CYC_C", "amount": 8000.0, "timestamp": "2026-10-09T02:10:00Z"},
            {"id": "E_C3", "transactionId": "TX_CYC_3", "source": "ACC_CYC_C", "target": "ACC_CYC_A", "amount": 8000.0, "timestamp": "2026-10-09T02:20:00Z"},
        ]
    }
    response = client.post("/predict/gnn", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "available"
    assert "cycle" in data["detected_topologies"]


def test_gnn_fan_in_topology_detection(client):
    registry.load_models_if_present()
    # Fan-in: Smurf1, Smurf2, Smurf3 all transfer to Aggregator
    payload = {
        "targetTransactionId": "TX_FAN_IN_1",
        "nodes": [
            {"id": "ACC_SMURF_1"},
            {"id": "ACC_SMURF_2"},
            {"id": "ACC_SMURF_3"},
            {"id": "ACC_AGGREGATOR"},
        ],
        "edges": [
            {"transactionId": "TX_FAN_IN_1", "source": "ACC_SMURF_1", "target": "ACC_AGGREGATOR", "amount": 9500.0, "timestamp": "2026-10-09T03:00:00Z"},
            {"transactionId": "TX_FAN_IN_2", "source": "ACC_SMURF_2", "target": "ACC_AGGREGATOR", "amount": 9200.0, "timestamp": "2026-10-09T03:05:00Z"},
            {"transactionId": "TX_FAN_IN_3", "source": "ACC_SMURF_3", "target": "ACC_AGGREGATOR", "amount": 9800.0, "timestamp": "2026-10-09T03:10:00Z"},
        ]
    }
    response = client.post("/predict/gnn", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "available"
    assert "fan-in" in data["detected_topologies"]


def test_gnn_empty_edge_set_handling(client):
    registry.load_models_if_present()
    payload = {
        "targetTransactionId": None,
        "nodes": [{"id": "ACC_SOLO_1"}, {"id": "ACC_SOLO_2"}],
        "edges": []
    }
    response = client.post("/predict/gnn", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "available"
    assert data["score"] == 0.0
    assert data["subgraph_edges_count"] == 0


def test_gnn_missing_checkpoint_returns_unavailable(client):
    original_instance = registry.models["gnn_detector"]["instance"]
    original_status = registry.models["gnn_detector"]["status"]
    try:
        registry.models["gnn_detector"]["status"] = "unavailable"
        registry.models["gnn_detector"]["instance"] = None

        payload = {
            "targetTransactionId": "TX_FAIL",
            "nodes": [{"id": "ACC_1"}, {"id": "ACC_2"}],
            "edges": [{"source": "ACC_1", "target": "ACC_2", "amount": 500.0, "timestamp": "2026-10-09T10:00:00Z"}]
        }
        response = client.post("/predict/gnn", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["score"] is None  # CRITICAL: must NOT be 0
    finally:
        registry.models["gnn_detector"]["instance"] = original_instance
        registry.models["gnn_detector"]["status"] = original_status
