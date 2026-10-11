import pytest
from pydantic import ValidationError
from app.schemas.transaction import TransactionPredictRequest
from app.schemas.anomaly import AnomalyPredictRequest
from app.schemas.gnn import GNNPredictRequest


def test_transaction_schema_aliases():
    # Test snake_case (Backend Person 1 style)
    req1 = TransactionPredictRequest(
        transaction_id="TX_1001",
        sender_id="ACC_001",
        receiver_id="ACC_002",
        amount=500.0,
        timestamp="2026-10-09T12:00:00Z",
        channel="online",
    )
    assert req1.transaction_id == "TX_1001"
    assert req1.sender_id == "ACC_001"
    assert req1.receiver_id == "ACC_002"

    # Test camelCase (Frontend Person 2/3 style)
    req2 = TransactionPredictRequest(
        id="TX_1002",
        senderAccountId="ACC_003",
        receiverAccountId="ACC_004",
        amount=1250.75,
        timestamp="2026-10-09T12:05:00Z",
    )
    assert req2.transaction_id == "TX_1002"
    assert req2.sender_id == "ACC_003"
    assert req2.receiver_id == "ACC_004"


def test_transaction_schema_validation_rejects_negative_amount():
    with pytest.raises(ValidationError):
        TransactionPredictRequest(
            transaction_id="TX_FAIL",
            sender_id="ACC_001",
            receiver_id="ACC_002",
            amount=-50.0,
            timestamp="2026-10-09T12:00:00Z",
        )


def test_anomaly_schema_validation():
    req = AnomalyPredictRequest(
        senderAccountId="ACC_BENCH",
        current_amount=2500.0,
        timestamp="2026-10-09T14:30:00Z",
        history=[
            {"amount": 100.0, "timestamp": "2026-10-08T10:00:00Z", "direction": "outbound"},
            {"amount": 150.0, "timestamp": "2026-10-08T18:00:00Z", "direction": "outbound"},
        ]
    )
    assert req.account_id == "ACC_BENCH"
    assert len(req.history) == 2


def test_gnn_schema_validation():
    req = GNNPredictRequest(
        targetTransactionId="TX_999",
        nodes=[{"id": "ACC_A"}, {"id": "ACC_B"}],
        edges=[{"source": "ACC_A", "target": "ACC_B", "amount": 900.0, "timestamp": "2026-10-09T12:00:00Z"}]
    )
    assert req.target_transaction_id == "TX_999"
    assert len(req.nodes) == 2
    assert len(req.edges) == 1
