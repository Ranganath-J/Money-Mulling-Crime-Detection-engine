import os
import sys
import json
import random
import math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.models.gnn_detector import GNNModelWrapper, MessagePassingGNN, NODE_FEAT_DIM, EDGE_FEAT_DIM
from app.config import settings


def generate_synthetic_graph_scenarios(num_scenarios: int = 120, seed: int = 42):
    """Generate reproducible synthetic graph scenarios for money muling detection.
    
    Includes:
    - Normal peer-to-peer / merchant bipartite graphs
    - Fan-in smurfing aggregators
    - Fan-out dispersal structures
    - Circular flow laundering / wash-trading cycles
    - Multi-hop pass-through chains
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    scenarios = []

    for s_idx in range(num_scenarios):
        # 60% normal graphs, 40% suspicious money muling graphs
        is_suspicious = (s_idx % 3 != 0)
        pattern_type = "normal"
        if is_suspicious:
            pattern_type = random.choice(["fan_in", "fan_out", "cycle", "layered_chain"])

        num_nodes = random.randint(8, 25)
        node_prefix = f"G{s_idx:03d}_ACC_"
        node_ids = [f"{node_prefix}{i:02d}" for i in range(num_nodes)]

        edges = []  # list of (src_idx, dst_idx, amount, hour, label)
        
        if pattern_type == "normal":
            # Normal sparse tree or random graph
            for _ in range(num_nodes * 2):
                u = random.randint(0, num_nodes - 1)
                v = random.randint(0, num_nodes - 1)
                if u != v:
                    amt = float(np.random.exponential(100.0) + 15.0)
                    hour = float(random.randint(8, 22))
                    edges.append((u, v, amt, hour, 0))

        elif pattern_type == "fan_in":
            # Multiple smurfs funnelling into single aggregator node 0
            aggregator = 0
            for u in range(1, min(7, num_nodes)):
                amt = float(np.random.uniform(8500.0, 9800.0))
                hour = float(random.randint(1, 5))  # unusual hours
                edges.append((u, aggregator, amt, hour, 1))
            # Background noise
            for _ in range(num_nodes):
                u = random.randint(0, num_nodes - 1)
                v = random.randint(0, num_nodes - 1)
                if u != v and u != aggregator and v != aggregator:
                    edges.append((u, v, float(np.random.exponential(50.0) + 10.0), 14.0, 0))

        elif pattern_type == "fan_out":
            # Disperser node 0 dispersing to multiple destination nodes
            disperser = 0
            for v in range(1, min(7, num_nodes)):
                amt = float(np.random.uniform(4000.0, 9500.0))
                hour = float(random.randint(2, 6))
                edges.append((disperser, v, amt, hour, 1))
            for _ in range(num_nodes):
                u = random.randint(1, num_nodes - 1)
                v = random.randint(1, num_nodes - 1)
                if u != v:
                    edges.append((u, v, float(np.random.exponential(60.0) + 10.0), 12.0, 0))

        elif pattern_type == "cycle":
            # Cycle 0 -> 1 -> 2 -> 3 -> 0
            cycle_len = min(4, num_nodes)
            amt = float(np.random.uniform(5000.0, 10000.0))
            for i in range(cycle_len):
                nxt = (i + 1) % cycle_len
                edges.append((i, nxt, amt, 3.0, 1))
            for _ in range(num_nodes):
                u = random.randint(cycle_len, num_nodes - 1) if num_nodes > cycle_len else 0
                v = random.randint(0, num_nodes - 1)
                if u != v:
                    edges.append((u, v, float(np.random.exponential(40.0) + 10.0), 15.0, 0))

        elif pattern_type == "layered_chain":
            # Linear pass-through chain 0 -> 1 -> 2 -> 3
            chain_len = min(5, num_nodes)
            amt = float(np.random.uniform(7000.0, 9500.0))
            for i in range(chain_len - 1):
                edges.append((i, i + 1, amt * 0.98, 2.0, 1))

        if not edges:
            edges.append((0, 1, 50.0, 12.0, 0))

        # Node features: compute inflow/outflow sums and degrees
        inflow = [0.0] * num_nodes
        outflow = [0.0] * num_nodes
        in_deg = [0] * num_nodes
        out_deg = [0] * num_nodes

        for u, v, amt, _, _ in edges:
            outflow[u] += amt
            inflow[v] += amt
            out_deg[u] += 1
            in_deg[v] += 1

        x_list = []
        for i in range(num_nodes):
            inf = inflow[i]
            outf = outflow[i]
            turnover = outf / (inf + 1e-4)
            is_mule = 1.0 if (pattern_type != "normal" and (in_deg[i] > 2 or out_deg[i] > 2)) else 0.0
            x_list.append([inf, outf, float(in_deg[i]), float(out_deg[i]), turnover, is_mule])

        x_tensor = torch.tensor(x_list, dtype=torch.float32)

        edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
        edge_attr = torch.tensor([
            [e[2], math.log1p(max(0.0, e[2])), e[3], 1.0, 1.0] for e in edges
        ], dtype=torch.float32)
        edge_y = torch.tensor([e[4] for e in edges], dtype=torch.float32)

        scenarios.append({
            "scenario_id": f"SCN_{s_idx:03d}",
            "pattern": pattern_type,
            "x": x_tensor,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_y": edge_y,
        })

    return scenarios


def train_gnn(seed: int = 42, epochs: int = 25):
    print(f"=== [Prompt 2A.3] Training PyTorch Geometric GNN (seed={seed}, epochs={epochs}) ===")
    
    # 1. Generate synthetic graph scenarios
    scenarios = generate_synthetic_graph_scenarios(num_scenarios=120, seed=seed)
    
    # 2. Leakage-aware graph-based split (70% train, 15% val, 15% test)
    # Graph-partitioned split guarantees that test topologies and node identities never appear in training.
    n = len(scenarios)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_graphs = scenarios[:train_end]
    val_graphs = scenarios[train_end:val_end]
    test_graphs = scenarios[val_end:]

    total_edges = sum(g["edge_y"].shape[0] for g in scenarios)
    pos_edges = sum(torch.sum(g["edge_y"]).item() for g in scenarios)
    neg_edges = total_edges - pos_edges
    pos_weight_val = neg_edges / (pos_edges + 1e-6)
    print(f"Total graph scenarios: {len(scenarios)} (Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)})")
    print(f"Total edges: {total_edges}, Suspicious mule edges: {int(pos_edges)}, Normal: {int(neg_edges)}")
    print(f"Class imbalance pos_weight: {pos_weight_val:.2f}")

    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)

    # 3. Model & Optimizer
    gnn_wrapper = GNNModelWrapper()
    model = gnn_wrapper.model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    # 4. Training loop
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        random.shuffle(train_graphs)
        for g in train_graphs:
            optimizer.zero_grad()
            _, logits = model(g["x"], g["edge_index"], g["edge_attr"])
            loss = F.binary_cross_entropy_with_logits(logits, g["edge_y"], pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"Epoch {ep+1:02d}/{epochs:02d} - Training Loss: {total_loss / len(train_graphs):.4f}")

    # 5. Evaluate on test graph scenarios
    print("\n--- Evaluating GNN on Graph-Partitioned Test Set ---")
    gnn_wrapper.is_fitted = True
    gnn_wrapper.test_data_cache = {"test_graphs": test_graphs}
    test_metrics = gnn_wrapper.evaluate(split="test", seed=seed)
    print("PyTorch Geometric GNN Test Metrics:")
    print(f"  Precision: {test_metrics.precision}")
    print(f"  Recall:    {test_metrics.recall}")
    print(f"  F1:        {test_metrics.f1}")
    print(f"  ROC-AUC:   {test_metrics.roc_auc}")
    print(f"  PR-AUC:    {test_metrics.pr_auc}")
    print(f"  FPR:       {test_metrics.false_positive_rate}")

    # 6. Save model checkpoint
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    gnn_wrapper.save_checkpoint(
        settings.gnn_model_path,
        extra_meta={
            "seed": seed,
            "epochs": epochs,
            "test_metrics": test_metrics.model_dump(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    print(f"\nSaved GNN checkpoint to: {settings.gnn_model_path}")

    # 7. Persist reproducible evaluation report
    eval_file = BASE_DIR / "app" / "evaluation" / "evaluation_metrics_2a3.json"
    with open(eval_file, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "split": "test",
            "split_strategy": "graph_partitioned_leakage_aware",
            "pos_weight": float(pos_weight_val),
            "gnn_metrics": test_metrics.model_dump(),
        }, f, indent=2)
    print(f"Saved evaluation metrics artifact to: {eval_file}")

    return test_metrics


if __name__ == "__main__":
    train_gnn(seed=42, epochs=25)
