import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import joblib

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.features.feature_extractor import TransactionFeatureExtractor, FEATURE_COLUMNS
from app.features.anomaly_features import AnomalyFeatureExtractor, ANOMALY_FEATURE_NAMES
from app.models.transaction_classifier import TransactionClassifierModel
from app.models.anomaly_detector import BehaviouralAnomalyDetectorModel
from app.config import settings


def generate_synthetic_transactions(n_samples: int = 5000, seed: int = 42):
    """Generate reproducible, realistic synthetic financial crime & normal transactions.
    
    Fixed seed ensures exact deterministic repeatability.
    """
    np.random.seed(seed)
    random.seed(seed)

    base_time = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    records = []
    labels = []

    accounts = [f"ACC_{i:04d}" for i in range(200)]
    merchants = [f"MERCH_{i:03d}" for i in range(50)]

    for i in range(n_samples):
        # 85% normal, 15% suspicious (class imbalance)
        is_suspicious = np.random.rand() < 0.15
        sender = random.choice(accounts)
        receiver = random.choice([a for a in accounts if a != sender])
        
        # Advance time monotonically to allow strict temporal splitting
        timestamp = base_time + timedelta(minutes=int(i * 5 + np.random.randint(0, 10)))
        
        if not is_suspicious:
            # Normal consumer spending patterns
            amount = float(np.random.exponential(scale=75.0) + 10.0)
            channel = random.choice(["online", "mobile", "pos", "pos", "online"])
            velocity = float(np.random.poisson(lam=1.0))
            hist_avg = float(amount * np.random.uniform(0.8, 1.2))
            hist_std = float(hist_avg * 0.3)
            inflow_outflow = float(np.random.uniform(0.7, 1.3))
            account_age = int(np.random.randint(60, 1200))
        else:
            # Money muling / structuring / rapid layering patterns
            mule_type = random.choice(["structuring", "burst_velocity", "unusual_timing", "pass_through"])
            if mule_type == "structuring":
                amount = float(np.random.uniform(8500.0, 9950.0))  # Just below $10,000 threshold
                velocity = float(np.random.randint(4, 12))
            elif mule_type == "burst_velocity":
                amount = float(np.random.uniform(500.0, 2500.0))
                velocity = float(np.random.randint(8, 20))
            elif mule_type == "pass_through":
                amount = float(np.random.uniform(3000.0, 7000.0))
                velocity = float(np.random.randint(3, 8))
            else:
                amount = float(np.random.uniform(4000.0, 9000.0))
                velocity = float(np.random.randint(2, 6))

            channel = random.choice(["wire", "online", "atm"])
            hist_avg = float(np.random.uniform(50.0, 200.0))  # History was low, now sudden spike!
            hist_std = float(hist_avg * 0.4)
            inflow_outflow = float(np.random.uniform(0.95, 1.05))  # Funds in quickly equal funds out
            account_age = int(np.random.randint(5, 45))  # Freshly opened mule accounts

        tx = {
            "transaction_id": f"TX_{i:06d}",
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": round(amount, 2),
            "currency": "USD",
            "timestamp": timestamp.isoformat(),
            "channel": channel,
            "merchant_id": random.choice(merchants) if np.random.rand() > 0.5 else None,
            "device_id": f"DEV_{random.randint(1, 150):03d}",
            "context": {
                "recent_velocity": velocity,
                "historical_avg_amount": round(hist_avg, 2),
                "historical_std_amount": round(hist_std, 2),
                "inflow_outflow_ratio": round(inflow_outflow, 2),
                "account_age_days": account_age,
            }
        }
        records.append(tx)
        labels.append(1 if is_suspicious else 0)

    return records, np.array(labels, dtype=int)


def run_training_pipeline(seed: int = 42):
    print(f"=== [Prompt 2A.2] Running Deterministic ML Training Pipeline (seed={seed}) ===")
    
    # 1. Generate synthetic dataset
    records, labels = generate_synthetic_transactions(n_samples=6000, seed=seed)
    
    # 2. Extract inference features (Strictly separating labels!)
    X_df = TransactionFeatureExtractor.extract_dataframe(records)
    X = X_df.values
    y = labels
    
    print(f"Extracted {X.shape[0]} samples with {X.shape[1]} features.")
    print(f"Class distribution: Normal={np.sum(y == 0)}, Suspicious={np.sum(y == 1)} ({np.mean(y)*100:.1f}%)")

    # 3. Leakage-aware temporal split (70% train, 15% validation, 15% test)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 4. Train Transaction Classifier: Logistic Regression Baseline
    print("\n--- Training Transaction Classifier Baseline (Logistic Regression) ---")
    lr_model = TransactionClassifierModel(model_type="logistic_regression", random_state=seed)
    lr_model.fit(X_train, y_train)
    lr_model.test_data_cache = {"X_test": X_test, "y_test": y_test}
    lr_metrics = lr_model.evaluate(split="test", seed=seed)
    print("Baseline Logistic Regression Test Metrics:")
    print(f"  Precision: {lr_metrics.precision}")
    print(f"  Recall:    {lr_metrics.recall}")
    print(f"  F1:        {lr_metrics.f1}")
    print(f"  ROC-AUC:   {lr_metrics.roc_auc}")
    print(f"  PR-AUC:    {lr_metrics.pr_auc}")
    print(f"  FPR:       {lr_metrics.false_positive_rate}")

    # Optionally compare with Random Forest
    print("\n--- Comparing with Random Forest Classifier ---")
    rf_model = TransactionClassifierModel(model_type="random_forest", random_state=seed)
    rf_model.fit(X_train, y_train)
    rf_model.test_data_cache = {"X_test": X_test, "y_test": y_test}
    rf_metrics = rf_model.evaluate(split="test", seed=seed)
    print("Random Forest Test Metrics:")
    print(f"  Precision: {rf_metrics.precision}")
    print(f"  Recall:    {rf_metrics.recall}")
    print(f"  F1:        {rf_metrics.f1}")
    print(f"  ROC-AUC:   {rf_metrics.roc_auc}")
    print(f"  PR-AUC:    {rf_metrics.pr_auc}")
    print(f"  FPR:       {rf_metrics.false_positive_rate}")

    # Use Random Forest as primary production model artifact (or Logistic Regression)
    selected_tx_model = rf_model if (rf_metrics.f1 or 0) >= (lr_metrics.f1 or 0) else lr_model

    # 5. Train Behavioural Anomaly Detector (Isolation Forest)
    print("\n--- Training Behavioural Anomaly Detector (Isolation Forest) ---")
    # Fit only on normal historical transactions to learn legitimate baseline
    X_norm_train = X_train[y_train == 0]
    
    # Create anomaly baseline features for fitting
    # [current_amount, hist_mean, amount_dev, freq_24h, hours_since, outbound_ratio]
    anom_train_feats = np.column_stack([
        X_norm_train[:, 0],  # amount
        X_norm_train[:, 0] * np.random.uniform(0.9, 1.1, size=len(X_norm_train)),  # hist_mean
        np.abs(X_norm_train[:, 7]),  # amount deviation
        X_norm_train[:, 6],  # frequency
        np.clip(24.0 / (X_norm_train[:, 6] + 0.1), 0.1, 72.0),  # hours since
        X_norm_train[:, 8],  # ratio
    ])

    anom_model = BehaviouralAnomalyDetectorModel(contamination=0.08, random_state=seed)
    anom_model.fit(anom_train_feats)

    anom_test_feats = np.column_stack([
        X_test[:, 0],
        X_test[:, 0] * np.random.uniform(0.9, 1.1, size=len(X_test)),
        np.abs(X_test[:, 7]),
        X_test[:, 6],
        np.clip(24.0 / (X_test[:, 6] + 0.1), 0.1, 72.0),
        X_test[:, 8],
    ])
    anom_model.test_data_cache = {"X_test": anom_test_feats, "y_test": y_test}
    anom_metrics = anom_model.evaluate(split="test", seed=seed)
    print("Isolation Forest Test Metrics:")
    print(f"  Precision: {anom_metrics.precision}")
    print(f"  Recall:    {anom_metrics.recall}")
    print(f"  F1:        {anom_metrics.f1}")
    print(f"  ROC-AUC:   {anom_metrics.roc_auc}")
    print(f"  PR-AUC:    {anom_metrics.pr_auc}")
    print(f"  FPR:       {anom_metrics.false_positive_rate}")

    # 6. Persist artifacts
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_tx_model, settings.transaction_model_path)
    joblib.dump(anom_model, settings.anomaly_model_path)
    print(f"\nSaved transaction classifier to: {settings.transaction_model_path}")
    print(f"Saved anomaly detector to:       {settings.anomaly_model_path}")

    # 7. Persist reproducible evaluation report
    eval_artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "split": "test",
        "dataset_size": len(X),
        "test_size": len(X_test),
        "baseline_logistic_regression": lr_metrics.model_dump(),
        "random_forest_classifier": rf_metrics.model_dump(),
        "behavioural_anomaly_isolation_forest": anom_metrics.model_dump(),
    }
    eval_file = BASE_DIR / "app" / "evaluation" / "evaluation_metrics_2a2.json"
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_file, "w") as f:
        json.dump(eval_artifact, f, indent=2)
    print(f"Saved evaluation metrics artifact to: {eval_file}")

    return {
        "lr_metrics": lr_metrics,
        "rf_metrics": rf_metrics,
        "anom_metrics": anom_metrics,
    }


if __name__ == "__main__":
    run_training_pipeline(seed=42)
