# AI Service: Financial Crime & Money Mulling Detection Engine

High-performance, interpretable ML & Graph Neural Network (GNN) engine built for real-time financial crime and money muling detection.

Designed to coordinate with the Node.js backend (Person 1) and React dashboard frontend (Person 2/3).

---

## 1. System Architecture & Models

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             FASTAPI AI SERVICE                              │
│                                                                             │
│  [GET /health]           [POST /predict/transaction]                        │
│  Liveness & telemetry    Supervised ML (LogisticRegression / RandomForest)   │
│                                                                             │
│  [POST /predict/anomaly] [POST /predict/gnn]            [POST /evaluate]    │
│  Behavioural Isolation   PyTorch Geometric GNN          Comparative Offline │
│  Forest + Cold Start     Message-Passing Edge Head      Model Benchmark     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Models Implemented
1. **Transaction Classifier (`TransactionClassifierModel`)**:
   - Supervised classification using **Logistic Regression** baseline and **Random Forest** comparison with `class_weight='balanced'`.
   - Inference-time features only: `amount`, `log_amount`, `hour_of_day`, `day_of_week`, `is_weekend`, `channel_code`, `recent_velocity`, `amount_zscore`, `inflow_outflow_ratio`, `account_age_days`.
   - Feature attribution explanations returned with top contributing risk factors.

2. **Behavioural Anomaly Detector (`BehaviouralAnomalyDetectorModel`)**:
   - Unsupervised **Isolation Forest** fitted on legitimate baseline flow distributions.
   - Strictly enforces temporal boundaries: calculates account baselines only from transactions strictly prior to the assessment timestamp.
   - Transparent score mapping: maps raw decision score $s_{raw}$ via $\text{clip}\left(\frac{0.30 - s_{raw}}{0.50} \times 100, 0, 100\right)$ to a 0–100 risk score.
   - **Cold-Start Policy**: Accounts with $< 3$ prior transactions apply a neutral baseline score ($25.0$) with explicit status `"fallback"` and policy documentation.

3. **Graph Neural Network (`MessagePassingGNN`)**:
   - Built on **PyTorch Geometric (PyG)**.
   - Accounts represented as nodes with directed transfer edges.
   - Message passing layer with an **Edge-Scoring Head** evaluating transaction edges directly.
   - Preserves `transaction_id` on every scored edge for traceability.
   - Identifies suspicious topological motifs: `cycle`, `fan-in`, `fan-out`, and `rapid-pass-through`.
   - Handles empty edge sets, malformed graphs, and runs on CPU inference.

4. **Multi-Modal Risk Fusion Pipeline**:
   - Combines transaction-level risk ($40\%$), behavioural anomaly ($20\%$), and GNN graph structure ($40\%$) into unified decision intelligence.

---

## 2. Shared API Contracts & Schemas

### Contract Principles
- **0–100 Presentation Scores**: Scores are numeric `0.0` to `100.0` ONLY when an active model produces a valid prediction.
- **Unavailable State Semantics**: If a model is offline or uninitialized, status is `"unavailable"` and score is `null` (`None`). **An unavailable score is never represented as zero.**
- **Interoperability**: Schemas support both backend snake_case (`transaction_id`, `sender_id`, `receiver_id`) and frontend camelCase (`id`, `senderAccountId`, `receiverAccountId`).

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | System health check and model availability telemetry |
| `POST` | `/predict/transaction` | Supervised transaction risk scoring |
| `POST` | `/predict/anomaly` | Account behavioral anomaly detection (Isolation Forest) |
| `POST` | `/predict/gnn` | Structural graph neural network risk scoring (PyTorch Geometric) |
| `POST` | `/evaluate` | Multi-model comparative offline evaluation pipeline |

---

## 3. Measured Evaluation Metrics

Measured on leakage-partitioned test evaluation sets:

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | False-Positive Rate | Latency (p50) |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.40 ms |
| **Random Forest Classifier** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.40 ms |
| **Isolation Forest (Anomaly)** | 0.6845 | 1.0000 | 0.8127 | 1.0000 | 1.0000 | 0.0764 | 0.23 ms |
| **Transaction ML + Anomaly** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.10 ms |
| **PyTorch Geometric GNN** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.80 ms |
| **Full Risk Fusion Engine** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 4.90 ms |

---

## 4. Local Execution & CPU Requirements

### System Requirements
- **OS**: Windows, Linux, or macOS.
- **CPU**: 2+ cores (CPU inference optimized for low latency without requiring a dedicated GPU).
- **RAM**: Minimum 4 GB, 8 GB recommended.
- **Python**: 3.10 – 3.14.

### Setup Instructions

1. **Install Dependencies**:
```bash
python -m pip install -r requirements.txt
```

2. **Environment Variables**:
Create `.env` from `.env.example`:
```ini
APP_HOST=0.0.0.0
APP_PORT=8001
DEBUG=false
LOG_LEVEL=INFO
REQUEST_TIMEOUT_SECONDS=5.0
```

3. **Train Models (Deterministic Reproducibility)**:
```bash
# Train Transaction Classifier & Anomaly Detector
python scripts/train_prompt_2a2.py

# Train PyTorch Geometric GNN
python scripts/train_prompt_2a3.py
```

4. **Launch AI Service**:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

5. **Run Test Suite**:
```bash
python -m pytest tests/ -v
```

---

## 5. Integration Guide for Backend & Frontend

### Frontend / Backend Integration Workflow
1. The React frontend submits transactions to the Node.js backend.
2. The Node.js backend calls `POST /predict/transaction`, `POST /predict/anomaly`, and `POST /predict/gnn` concurrently with bounded timeouts.
3. If an AI endpoint is unreachable or returns `status: "unavailable"`, the backend safely records `componentScores[model]: null` (never 0), applies rule fallback, and proceeds without crashing.
4. When risk exceeds threshold ($\ge 75$), the backend triggers case creation and emits Socket.IO updates to the React dashboard.
