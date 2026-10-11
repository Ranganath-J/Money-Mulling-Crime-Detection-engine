# Financial Crime & Money Mulling Detection Engine: AI Service

An interpretable, real-time Machine Learning (ML) and Graph Neural Network (GNN) detection engine for financial crime and money muling networks. 

This repository branch (`feature/ml-service`) implements the complete **Person 2A Workstream** (Prompts 2A.1 through 2A.4), designed to synchronously serve inference signals to the Node.js/Express backend (Person 1) and feed risk intelligence into the React/Vite dashboard frontend (`feature/frontend-dashboard`).

---

## Table of Contents

1. [Executive Summary & High-Level Architecture](#1-executive-summary--high-level-architecture)
2. [Complete File-by-File Directory Breakdown](#2-complete-file-by-file-directory-breakdown)
   - [Root Directory Configuration](#root-directory-configuration)
   - [AI Service Application (`ai-service/app/`)](#ai-service-application-ai-serviceapp)
   - [API Endpoints (`app/api/`)](#api-endpoints-appapi)
   - [Pydantic Schemas & Data Contracts (`app/schemas/`)](#pydantic-schemas--data-contracts-appschemas)
   - [Feature Engineering (`app/features/`)](#feature-engineering-appfeatures)
   - [Detection Models & Checkpoints (`app/models/`)](#detection-models--checkpoints-appmodels)
   - [Evaluation Pipeline (`app/evaluation/`)](#evaluation-pipeline-appevaluation)
   - [Service Management (`app/services/`)](#service-management-appservices)
   - [Deterministic Training Scripts (`scripts/`)](#deterministic-training-scripts-scripts)
   - [Automated Test Suite (`tests/`)](#automated-test-suite-tests)
3. [Contract Rules & Frontend Interoperability](#3-contract-rules--frontend-interoperability)
4. [Prompt Requirements & Implementation Matrix](#4-prompt-requirements--implementation-matrix)
5. [Measured Model Benchmark Results](#5-measured-model-benchmark-results)
6. [How to Run, Test, and Verify Locally](#6-how-to-run-test-and-verify-locally)
7. [API Endpoint Usage Examples](#7-api-endpoint-usage-examples)

---

## 1. Executive Summary & High-Level Architecture

The AI service operates as a high-performance Python FastAPI microservice that analyzes transactions, account histories, and graph topologies through four distinct layers of risk intelligence:

```
                                  INCOMING TRANSACTION / GRAPH
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐                  ┌──────────────────┐
│  Transaction ML  │                  │Behavioural Anom. │                  │     PyG GNN      │
│  Classifier      │                  │Detector          │                  │  Message Passing │
│                  │                  │                  │                  │                  │
│ • Supervised LR  │                  │ • Isolation      │                  │ • Node Embeddings│
│   & Random Forest│                  │   Forest         │                  │ • Edge Scoring   │
│ • Z-scores, flow │                  │ • Historical     │                  │ • Topologies:    │
│   velocity ratios│                  │   baselining     │                  │   cycles, fan-in,│
│ • Feature weights│                  │ • Cold-start     │                  │   fan-out, chains│
│   & explanations │                  │   handling       │                  │ • Edge-level risk│
└────────┬─────────┘                  └────────┬─────────┘                  └────────┬─────────┘
         │                                     │                                     │
         └─────────────────────────────────────┼─────────────────────────────────────┘
                                               ▼
                               ┌──────────────────────────────┐
                               │  Multi-Modal Risk Fusion     │
                               │  (Weighted Signal Synthesis) │
                               └──────────────┬───────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
           [Backend Node.js API]                            [React Dashboard UI]
         Synchronous HTTP Assessment                      Live Component Scores,
       (Timeout-safe, Bounded Payloads)                 Top Reasons, Graph Visuals
```

---

## 2. Complete File-by-File Directory Breakdown

### Root Directory Configuration

| File | Purpose & Responsibilities |
|---|---|
| [`README.md`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/README.md) | **Master Documentation**: Comprehensive overview of the system, complete file-by-file explanations, architecture diagrams, training instructions, and API contracts. |
| [`.gitignore`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/.gitignore) | **Git Ignore Manifest**: Prevents temporary files, caches (`__pycache__`, `.pytest_cache`, `.coverage`), environment configs (`.env`), virtual environments (`.venv`), and `node_modules` from cluttering source control. |

---

### AI Service Application (`ai-service/app/`)

| File | Purpose & Responsibilities |
|---|---|
| [`ai-service/requirements.txt`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/requirements.txt) | **Dependencies**: Explicit list of required Python libraries including FastAPI, Uvicorn, Scikit-learn, PyTorch, PyTorch Geometric, Pandas, NumPy, Pydantic, and Pytest. |
| [`ai-service/.env.example`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/.env.example) | **Environment Template**: Example configuration for host, port (`8001`), debug flags, timeout thresholds (`5.0s`), logging levels, and artifact storage directories. |
| [`ai-service/README.md`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/README.md) | **Microservice Guide**: Focused local run instructions, benchmark summaries, and endpoint routing reference for the AI service. |
| [`ai-service/app/__init__.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/__init__.py) | **App Package**: Marks `app` as a Python package and exposes package version metadata (`1.0.0`). |
| [`ai-service/app/main.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/main.py) | **FastAPI Application Entrypoint**: Initializes the application, registers CORS middleware for frontend communication, configures request timing headers (`X-Process-Time-Ms`), binds centralized structured logging, establishes startup/shutdown lifecycles, and installs global exception handlers. |
| [`ai-service/app/config.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/config.py) | **Settings & Environment Loader**: Implements Pydantic `BaseSettings` to safely load configuration from `.env` or system variables without hard-coded secrets. Manages artifact paths, timeout bounds, and maximum request constraints. |
| [`ai-service/app/logging.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/logging.py) | **Structured JSON Observability**: Formats application logs into structured JSON containing timestamps (UTC), log levels, modules, function names, line numbers, and request execution durations for automated log aggregation. |

---

### API Endpoints (`app/api/`)

| File | Purpose & Responsibilities |
|---|---|
| [`ai-service/app/api/__init__.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/api/__init__.py) | **Router Registry**: Aggregates all endpoint routers (`health_router`, `predict_router`, `evaluate_router`) into a unified `api_router` mounted onto the main FastAPI application. |
| [`ai-service/app/api/health.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/api/health.py) | **Health & Telemetry (`GET /health`)**: Returns service health status (`ok`, `degraded`, `unavailable`), environment name, uptime timestamp, and status for each individual model (`transaction_classifier`, `anomaly_detector`, `gnn_detector`). |
| [`ai-service/app/api/predict.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/api/predict.py) | **Inference Endpoints**: Houses the three core prediction routes: <br>• `POST /predict/transaction`: Evaluates single transactions via supervised ML. <br>• `POST /predict/anomaly`: Evaluates account behavioral shifts via Isolation Forest. <br>• `POST /predict/gnn`: Evaluates subgraphs via PyTorch Geometric. <br>Ensures that uninitialized or missing models return an explicit `status="unavailable"` and `score=null` (never 0). |
| [`ai-service/app/api/evaluate.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/api/evaluate.py) | **Evaluation Endpoint (`POST /evaluate`)**: Executes an on-demand, reproducible evaluation benchmark across all models and multi-modal combinations, returning precision, recall, F1, ROC-AUC, PR-AUC, false-positive rate, and latency. |

---

### Pydantic Schemas & Data Contracts (`app/schemas/`)

| File | Purpose & Responsibilities |
|---|---|
| [`ai-service/app/schemas/__init__.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/schemas/__init__.py) | **Schema Package Export**: Centralized import location for all request/response schemas. |
| [`ai-service/app/schemas/common.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/schemas/common.py) | **Core Contracts**: Defines base types shared across all endpoints: <br>• `BasePredictionResponse`: Core fields (`model_name`, `model_version`, `status`, `score`, `explanation`, `evidence`, `reasons`, `latency_ms`, `error`). <br>• `HealthResponse`: System status and model health breakdown. <br>• `ModelStatus`: Literal typing (`"available"`, `"unavailable"`, `"fallback"`). |
| [`ai-service/app/schemas/transaction.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/schemas/transaction.py) | **Transaction Contracts**: <br>• `TransactionPredictRequest`: Supports both backend snake_case (`transaction_id`, `sender_id`, `receiver_id`) and frontend camelCase (`id`, `senderAccountId`, `receiverAccountId`). Implements bounded validation ($0 < \text{amount} \le \$10,000,000$). <br>• `TransactionPredictResponse`: Returns the presentation score (0–100), risk category (`'Monitor'`, `'Review'`, `'Investigation Alert'`), and feature attribution explanation. |
| [`ai-service/app/schemas/anomaly.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/schemas/anomaly.py) | **Anomaly Contracts**: <br>• `AccountHistoryItem`: Historical transaction item prior to assessment cutoff. <br>• `AnomalyPredictRequest`: Current transaction plus account history list (bounded to 1,000 items). <br>• `AnomalyPredictResponse`: Raw Isolation Forest decision score, mapped 0–100 presentation score, cold-start flag (`is_cold_start`), and cold-start policy details. |
| [`ai-service/app/schemas/gnn.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/schemas/gnn.py) | **Graph Contracts**: <br>• `GraphNodePayload`: Account node attributes (inflow, outflow, degree, candidate flags). Bounded up to 5,000 nodes. <br>• `GraphEdgePayload`: Directed money transfer attributes (source, target, amount, timestamp). Bounded up to 20,000 edges. <br>• `GNNPredictResponse`: Ego-subgraph risk score, list of individual `EdgeRiskScore` objects with preserved transaction IDs, and detected topological structures (`cycle`, `fan-in`, `fan-out`, `rapid-pass-through`). |
| [`ai-service/app/schemas/evaluation.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/schemas/evaluation.py) | **Benchmark Contracts**: Defines `EvaluationRequest`, `EvaluationMetrics` (precision, recall, F1, ROC-AUC, PR-AUC, FPR, p50/p95 latency), and `EvaluationResponse`. |

---

### Feature Engineering (`app/features/`)

| File | Purpose & Responsibilities |
|---|---|
| [`ai-service/app/features/__init__.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/features/__init__.py) | **Features Package**: Exports feature extraction utilities. |
| [`ai-service/app/features/feature_extractor.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/features/feature_extractor.py) | **Transaction Feature Extractor**: Transforms raw transaction payloads into 10 numeric features: `amount`, `log_amount`, `hour_of_day`, `day_of_week`, `is_weekend`, `channel_code`, `recent_velocity`, `amount_zscore`, `inflow_outflow_ratio`, and `account_age_days`. **Strictly excludes labels** to guarantee zero training-time leakage. |
| [`ai-service/app/features/anomaly_features.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/features/anomaly_features.py) | **Behavioral Baseline Extractor**: Calculates account behavioral profiles strictly from events timestamped prior to the assessment moment. Computes historical mean amount, amount deviation ratios, 24h rolling velocity, and hours elapsed since last activity. Implements cold-start detection (< 3 events). |

---

### Detection Models & Checkpoints (`app/models/`)

| File | Purpose & Responsibilities |
|---|---|
| [`ai-service/app/models/__init__.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/models/__init__.py) | **Models Package**: Exports model wrapper classes. |
| [`ai-service/app/models/transaction_classifier.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/models/transaction_classifier.py) | **Supervised Classifier**: Wraps a Scikit-learn `Pipeline` with `StandardScaler` and `LogisticRegression(class_weight='balanced')` (interpretable baseline) alongside a `RandomForestClassifier`. Computes feature attribution explanations and outputs calibrated probabilities mapped to 0–100 presentation scores. |
| [`ai-service/app/models/anomaly_detector.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/models/anomaly_detector.py) | **Unsupervised Isolation Forest**: Identifies statistical outliers from normal behavioral profiles. Implements the mathematical mapping function $\text{score} = \text{clip}\left(\frac{0.30 - s_{raw}}{0.50} \times 100, 0, 100\right)$ and enforces the cold-start fallback policy (returning score 25.0 and status `"fallback"` for new accounts). |
| [`ai-service/app/models/gnn_detector.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/models/gnn_detector.py) | **PyTorch Geometric GNN**: Defines `MessagePassingGNN` and `EdgeScoringHead`. Propagates account node features across directed transfer edges, evaluates edge logits, preserves `transaction_id` on each edge score, and runs graph topology traversal algorithms to detect cycles, fan-in, fan-out, and pass-through paths. |
| [`ai-service/app/models/artifacts/transaction_classifier.joblib`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/models/artifacts/transaction_classifier.joblib) | **Saved ML Pipeline Checkpoint**: Serialized joblib artifact containing the fitted preprocessor and classifier weights. |
| [`ai-service/app/models/artifacts/anomaly_detector.joblib`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/models/artifacts/anomaly_detector.joblib) | **Saved Anomaly Model Checkpoint**: Serialized joblib artifact containing the fitted Isolation Forest trees. |
| [`ai-service/app/models/artifacts/gnn_detector.pt`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/models/artifacts/gnn_detector.pt) | **Saved PyTorch GNN Checkpoint**: Serialized PyTorch state dictionary, layer specifications, and benchmark metadata. |

---

### Evaluation Pipeline (`app/evaluation/`)

| File | Purpose & Responsibilities |
|---|---|
| [`ai-service/app/evaluation/__init__.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/evaluation/__init__.py) | **Evaluation Package**: Exports benchmark pipeline runners. |
| [`ai-service/app/evaluation/pipeline.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/evaluation/pipeline.py) | **Comparative Evaluation Pipeline**: Runs an automated 5-way benchmark comparing: <br>1. Transaction Classifier only <br>2. Behavioral Anomaly Detector only <br>3. Transaction ML + Anomaly ensemble <br>4. Graph Neural Network only <br>5. Full Multi-Modal Risk Fusion engine <br>Calculates precision, recall, F1, ROC-AUC, PR-AUC, false-positive rate, and p50/p95 latency. |
| [`ai-service/app/evaluation/evaluation_metrics_2a2.json`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/evaluation/evaluation_metrics_2a2.json) | **Prompt 2A.2 Benchmark Artifact**: Reproducible metrics for Logistic Regression, Random Forest, and Isolation Forest. |
| [`ai-service/app/evaluation/evaluation_metrics_2a3.json`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/evaluation/evaluation_metrics_2a3.json) | **Prompt 2A.3 Benchmark Artifact**: Reproducible metrics for the PyTorch Geometric GNN on graph-partitioned test sets. |
| [`ai-service/app/evaluation/evaluation_report_full.json`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/evaluation/evaluation_report_full.json) | **Prompt 2A.4 Full Report Artifact**: Multi-model comparison report documenting model versions, schema versions, random seeds, and evaluation configurations. |

---

### Service Management (`app/services/`)

| File | Purpose & Responsibilities |
|---|---|
| [`ai-service/app/services/__init__.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/services/__init__.py) | **Services Package**: Exports model registry services. |
| [`ai-service/app/services/model_registry.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/app/services/model_registry.py) | **Model Lifecycle & Health Registry**: Singleton registry managing model discovery, loading from disk, health monitoring, and unavailable state handling. Ensures uninitialized models return `None` (never 0) for scores. |

---

### Deterministic Training Scripts (`scripts/`)

| File | Purpose & Responsibilities |
|---|---|
| [`ai-service/scripts/train_prompt_2a2.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/scripts/train_prompt_2a2.py) | **ML & Anomaly Trainer**: Generates 6,000 synthetic transaction records with fixed seed (42), applies a 70/15/15 temporal split, fits Logistic Regression and Random Forest models, fits the Isolation Forest baseline, and persists joblib artifacts to `app/models/artifacts/`. |
| [`ai-service/scripts/train_prompt_2a3.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/scripts/train_prompt_2a3.py) | **GNN Trainer**: Generates 120 graph scenarios (smurfing fan-in, dispersal fan-out, circular layering cycles, multi-hop pass-through chains), applies leakage-aware graph partitioning, trains `MessagePassingGNN` using weighted BCE loss (`pos_weight=5.0`), and saves checkpoint to `app/models/artifacts/gnn_detector.pt`. |

---

### Automated Test Suite (`tests/`)

| File | Purpose & Responsibilities |
|---|---|
| [`ai-service/tests/__init__.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/tests/__init__.py) | **Test Package**: Marks tests as a Python package. |
| [`ai-service/tests/conftest.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/tests/conftest.py) | **Pytest Fixtures**: Initializes FastAPI `TestClient` and resets model registry states between test runs. |
| [`ai-service/tests/test_health.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/tests/test_health.py) | **Health Tests**: Validates `GET /health` response shape, status codes, and model availability flags. |
| [`ai-service/tests/test_schemas.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/tests/test_schemas.py) | **Schema Tests**: Verifies Pydantic alias handling (camelCase vs snake_case) and input validation (e.g., rejecting negative amounts). |
| [`ai-service/tests/test_contracts.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/tests/test_contracts.py) | **Contract Tests**: Enforces Prompt 2A.1 contract rules: ensures unavailable models return `score=None` (never 0), verifies fallback statuses, and tests HTTP 422 responses on malformed payloads. |
| [`ai-service/tests/test_prompt_2a2.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/tests/test_prompt_2a2.py) | **ML & Anomaly Tests**: Tests normal spending, high-velocity mule transactions, cold-start fallback policy, established account deviations, class imbalance handling, and deterministic repeatability. |
| [`ai-service/tests/test_prompt_2a3.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/tests/test_prompt_2a3.py) | **GNN Tests**: Tests graph inference, cycle pattern detection, smurfing fan-in detection, empty edge set handling, and missing checkpoint handling. |
| [`ai-service/tests/test_prompt_2a4.py`](file:///c:/Users/Admin/Desktop/Money-Mulling-Crime-Detection-engine/ai-service/tests/test_prompt_2a4.py) | **Hardening Tests**: Verifies bounded payload constraints (rejecting amounts > $10M, empty nodes), tests the multi-model offline evaluation pipeline, and tests exception safety. |

---

## 3. Contract Rules & Frontend Interoperability

The service aligns with both the Node.js backend (`origin/feature/backend`) and the React dashboard (`feature/frontend-dashboard`):

1. **Dual-Key Compatibility**:
   - Backend snake_case: `transaction_id`, `sender_id`, `receiver_id`, `amount`, `timestamp`.
   - Frontend camelCase: `id`, `senderAccountId`, `receiverAccountId`, `amount`, `timestamp`.
   - Pydantic models automatically accept either format.

2. **0–100 Score Semantics**:
   - Valid scores are always numeric floats in $[0.0, 100.0]$.
   - When a model is offline or uninitialized, `status` is `"unavailable"` and `score` is `null` (`None`). **An unavailable score is never represented as 0.**

3. **Risk Categorization**:
   - Matches frontend `RiskCategory`:
     - `'Monitor'`: Score $< 40.0$
     - `'Review'`: Score $40.0 \le \text{score} < 75.0$
     - `'Investigation Alert'`: Score $\ge 75.0$

4. **Component Scores**:
   - Returns individual component scores that populate the frontend `componentScores` map:
     - `transaction_classifier`: Supervised ML score
     - `anomaly_detector`: Behavioral Isolation Forest score
     - `gnn_detector`: Topological graph score

---

## 4. Prompt Requirements & Implementation Matrix

| Requirement | Implementation Details | Status |
|---|---|:---:|
| **Prompt 2A.1**: Establish FastAPI Service & Contract | Created stable FastAPI service with `/health`, `/predict/transaction`, `/predict/anomaly`, `/predict/gnn`, `/evaluate`. Uninitialized models return explicit `"unavailable"` status and `null` scores. | Complete |
| **Prompt 2A.2**: Supervised ML & Behavioral Anomaly | Logistic Regression baseline + Random Forest with balanced class weights. Isolation Forest with documented mathematical mapping and cold-start fallback policy (< 3 events). | Complete |
| **Prompt 2A.3**: PyTorch Geometric GNN | `MessagePassingGNN` with directed transfers as edges, accounts as nodes, edge scoring head preserving `transaction_id`, and topology detection (cycles, fan-in, fan-out, chains). | Complete |
| **Prompt 2A.4**: Evaluation Pipeline & Hardening | 5-way comparative evaluation pipeline, bounded payload validation (amounts $\le \$10\text{M}$, nodes $\le 5000$, edges $\le 20000$), exception safety, and structured JSON logs. | Complete |

---

## 5. Measured Model Benchmark Results

Measured on strictly partitioned, leakage-free test sets:

| Model Architecture | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | False-Positive Rate | p50 Latency |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.40 ms |
| **Random Forest Classifier** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.40 ms |
| **Isolation Forest (Anomaly)** | 0.6845 | 1.0000 | 0.8127 | 1.0000 | 1.0000 | 0.0764 | 0.23 ms |
| **Transaction ML + Anomaly** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.10 ms |
| **PyTorch Geometric GNN** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.80 ms |
| **Full Risk Fusion Engine** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 4.90 ms |

---

## 6. How to Run, Test, and Verify Locally

### Step 1: Activate Virtual Environment

**In PowerShell (from repository root):**
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Or in Command Prompt: `.venv\Scripts\activate.bat` | Git Bash: `source .venv/Scripts/activate`)*

### Step 2: Run Automated Tests

Navigate to `ai-service/` and execute Pytest:
```powershell
cd ai-service
pytest tests/ -v
```
**Actual Result**: All 26 tests pass in under 2 seconds.

### Step 3: Start the Microservice

```powershell
cd ai-service
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Step 4: Access in Your Browser

- **Interactive Swagger Docs UI**: [http://localhost:8001/docs](http://localhost:8001/docs)
- **Health Telemetry**: [http://localhost:8001/health](http://localhost:8001/health)
- **ReDoc UI**: [http://localhost:8001/redoc](http://localhost:8001/redoc)

*(Note: Always use `localhost:8001` or `127.0.0.1:8001` in your browser. Browsers will reject `0.0.0.0:8001` with `ERR_ADDRESS_INVALID`.)*

### Step 5: (Optional) Retrain Models Deterministically

To retrain from synthetic generators:
```powershell
python scripts/train_prompt_2a2.py
python scripts/train_prompt_2a3.py
```

---

## 7. API Endpoint Usage Examples

### 1. Predict Transaction Risk (`POST /predict/transaction`)

**Request:**
```json
{
  "id": "TX_9901",
  "senderAccountId": "ACC_MULE_01",
  "receiverAccountId": "ACC_DEST_02",
  "amount": 9500.0,
  "timestamp": "2026-10-10T14:30:00Z",
  "channel": "wire",
  "context": {
    "recent_velocity": 12.0,
    "historical_avg_amount": 60.0,
    "historical_std_amount": 15.0,
    "inflow_outflow_ratio": 1.0,
    "account_age_days": 10
  }
}
```

**Response:**
```json
{
  "model_name": "TransactionClassifier",
  "model_version": "1.0.0",
  "status": "available",
  "score": 100.0,
  "risk_category": "Investigation Alert",
  "transaction_id": "TX_9901",
  "confidence": 1.0,
  "reasons": [
    "High monetary value transfer",
    "Elevated transaction frequency within recent rolling window",
    "Transaction amount deviates significantly from account baseline"
  ],
  "latency_ms": 1.15
}
```

---

### 2. Behavioral Anomaly Detection (`POST /predict/anomaly`)

**Request (Cold-Start Example):**
```json
{
  "senderAccountId": "ACC_NEW_USER",
  "current_amount": 500.0,
  "timestamp": "2026-10-10T12:00:00Z",
  "history": []
}
```

**Response:**
```json
{
  "model_name": "BehaviouralAnomalyDetector",
  "model_version": "1.0.0",
  "status": "fallback",
  "score": 25.0,
  "account_id": "ACC_NEW_USER",
  "is_cold_start": true,
  "cold_start_policy_applied": "Cold-start neutral score (25.0) applied: Account history < 3 events. Baselining in progress.",
  "reasons": [
    "Account has insufficient history for behavioral profiling (cold start)"
  ]
}
```

---

### 3. Graph Neural Network Topology Detection (`POST /predict/gnn`)

**Request:**
```json
{
  "targetTransactionId": "TX_CYCLE_1",
  "nodes": [
    {"id": "ACC_A", "inflow_sum": 8000.0, "outflow_sum": 8000.0, "degree": 2},
    {"id": "ACC_B", "inflow_sum": 8000.0, "outflow_sum": 8000.0, "degree": 2},
    {"id": "ACC_C", "inflow_sum": 8000.0, "outflow_sum": 8000.0, "degree": 2}
  ],
  "edges": [
    {"transactionId": "TX_CYCLE_1", "source": "ACC_A", "target": "ACC_B", "amount": 8000.0, "timestamp": "2026-10-10T02:00:00Z"},
    {"transactionId": "TX_CYCLE_2", "source": "ACC_B", "target": "ACC_C", "amount": 8000.0, "timestamp": "2026-10-10T02:10:00Z"},
    {"transactionId": "TX_CYCLE_3", "source": "ACC_C", "target": "ACC_A", "amount": 8000.0, "timestamp": "2026-10-10T02:20:00Z"}
  ]
}
```

**Response:**
```json
{
  "model_name": "PyTorchGeometricGNN",
  "model_version": "1.0.0",
  "status": "available",
  "score": 100.0,
  "transaction_id": "TX_CYCLE_1",
  "subgraph_nodes_count": 3,
  "subgraph_edges_count": 3,
  "detected_topologies": ["cycle", "rapid-pass-through"],
  "edge_scores": [
    {
      "transaction_id": "TX_CYCLE_1",
      "edge_id": "edge_0",
      "source": "ACC_A",
      "target": "ACC_B",
      "amount": 8000.0,
      "score": 100.0,
      "pattern": "cycle"
    }
  ],
  "reasons": [
    "Network structure exhibits topologies: cycle, rapid-pass-through",
    "High edge probability of money muling flow identified by message-passing GNN"
  ]
}
```