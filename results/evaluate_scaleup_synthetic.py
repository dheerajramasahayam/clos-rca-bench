from __future__ import annotations

import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from root_cause_analysis.topology_rca_model import SpatioTemporalGraphModel


RESULTS_DIR = Path("results")
GRAPHS_DIR = Path("graphs")

CAUSE_NAMES = {
    1: "link_failure",
    2: "bgp_issue",
    3: "ecmp_shift",
    4: "interface_shutdown",
}
CAUSE_PATTERNS = {
    1: np.array([1.10, 0.30, 0.60, 0.90, 0.10, 0.00, 1.20, 0.20], dtype=np.float32),
    2: np.array([0.20, 0.50, 1.00, 0.80, 1.10, 0.50, 0.20, 0.10], dtype=np.float32),
    3: np.array([0.20, 1.10, 0.70, 0.40, 0.90, 1.00, 0.10, 0.20], dtype=np.float32),
    4: np.array([0.80, 0.80, 0.90, 0.80, 0.20, 0.20, 0.30, 1.10], dtype=np.float32),
}

SEQ_LEN = 6
FEATURE_DIM = 8
WINDOW_COUNT = 900
GRAPH_CONFIG = {
    "topology_name": "Synthetic Clos-59",
    "core_count": 5,
    "spine_count": 9,
    "leaf_count": 45,
}


@dataclass(frozen=True)
class SyntheticBundle:
    X: np.ndarray
    y_anomaly: np.ndarray
    y_cause: np.ndarray
    y_target: np.ndarray
    adjacency: np.ndarray
    node_names: list[str]
    metadata: pd.DataFrame


def build_clos_topology(core_count: int, spine_count: int, leaf_count: int):
    node_names = [
        *(f"core{i + 1}" for i in range(core_count)),
        *(f"spine{i + 1}" for i in range(spine_count)),
        *(f"leaf{i + 1}" for i in range(leaf_count)),
    ]
    node_count = len(node_names)
    adjacency = np.eye(node_count, dtype=np.float32)

    core_indices = list(range(core_count))
    spine_indices = list(range(core_count, core_count + spine_count))
    leaf_indices = list(range(core_count + spine_count, node_count))

    for core_idx in core_indices:
        for spine_idx in spine_indices:
            adjacency[core_idx, spine_idx] = 1.0
            adjacency[spine_idx, core_idx] = 1.0

    for spine_idx in spine_indices:
        for leaf_idx in leaf_indices:
            adjacency[spine_idx, leaf_idx] = 1.0
            adjacency[leaf_idx, spine_idx] = 1.0

    degree = adjacency.sum(axis=1)
    adjacency = adjacency / np.sqrt(np.outer(degree, degree))
    return node_names, adjacency, core_indices, spine_indices, leaf_indices


def shortest_path_matrix(adjacency: np.ndarray) -> np.ndarray:
    binary = (adjacency > 0).astype(int)
    node_count = adjacency.shape[0]
    distances = np.full((node_count, node_count), 99, dtype=int)

    for source in range(node_count):
        distances[source, source] = 0
        queue = deque([source])
        seen = {source}
        while queue:
            current = queue.popleft()
            for neighbor in np.where(binary[current] > 0)[0]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                distances[source, neighbor] = distances[source, current] + 1
                queue.append(neighbor)

    return distances


def generate_scaleup_dataset(
    sample_count: int = WINDOW_COUNT,
    seq_len: int = SEQ_LEN,
    feature_dim: int = FEATURE_DIM,
    seed: int = 42,
) -> SyntheticBundle:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    node_names, adjacency, core_indices, spine_indices, leaf_indices = build_clos_topology(
        core_count=GRAPH_CONFIG["core_count"],
        spine_count=GRAPH_CONFIG["spine_count"],
        leaf_count=GRAPH_CONFIG["leaf_count"],
    )
    distances = shortest_path_matrix(adjacency)
    hidden_target_indices = set(core_indices[::2] + leaf_indices[::3])

    X = np.random.normal(0.0, 0.05, size=(sample_count, seq_len, len(node_names), feature_dim)).astype(np.float32)
    y_anomaly = np.zeros(sample_count, dtype=int)
    y_cause = np.zeros(sample_count, dtype=int)
    y_target = np.zeros(sample_count, dtype=int)
    metadata_rows = []

    for sample_idx in range(sample_count):
        random_draw = np.random.rand()
        if random_draw < 0.28:
            metadata_rows.append(
                {
                    "window_id": sample_idx,
                    "fault_mode": "normal",
                    "hidden_target": False,
                    "primary_cause": "normal",
                    "primary_target": "none",
                }
            )
            continue

        y_anomaly[sample_idx] = 1
        simultaneous = random_draw > 0.78
        fault_mode = "simultaneous" if simultaneous else "single"
        fault_count = 2 if simultaneous else 1
        used_targets: set[int] = set()
        primary_cause = 0
        primary_target_idx = 0
        primary_hidden = False

        for fault_index in range(fault_count):
            cause_id = int(np.random.randint(1, 5))
            if cause_id == 1:
                candidates = leaf_indices + spine_indices
            elif cause_id == 2:
                candidates = core_indices + spine_indices
            elif cause_id == 3:
                candidates = spine_indices + core_indices + leaf_indices[::4]
            else:
                candidates = leaf_indices + spine_indices

            candidates = [candidate for candidate in candidates if candidate not in used_targets]
            target_idx = int(np.random.choice(candidates))
            used_targets.add(target_idx)
            onset = int(np.random.randint(1, 3) + fault_index)
            pattern = CAUSE_PATTERNS[cause_id] * np.random.uniform(0.9, 1.15)
            root_hidden = target_idx in hidden_target_indices

            if fault_index == 0:
                primary_cause = cause_id
                primary_target_idx = target_idx
                primary_hidden = root_hidden
                y_cause[sample_idx] = cause_id
                y_target[sample_idx] = target_idx + 1

            for node_idx in range(len(node_names)):
                distance = distances[target_idx, node_idx]
                if distance >= 99:
                    continue

                amplitude = 0.72 ** distance
                if node_idx == target_idx and root_hidden:
                    amplitude *= 0.08
                elif node_idx == target_idx:
                    amplitude *= 1.2
                elif root_hidden and distance == 1:
                    amplitude *= 1.3

                topology_factor = 1.25 if (node_idx in leaf_indices and target_idx in leaf_indices and distance == 2) else 1.0
                for time_idx in range(onset, seq_len):
                    time_gain = min(1.0, 0.42 * (time_idx - onset + 1))
                    X[sample_idx, time_idx, node_idx] += pattern * amplitude * topology_factor * time_gain

            for neighbor_idx in np.where(adjacency[target_idx] > 0)[0]:
                if neighbor_idx == target_idx:
                    continue
                for time_idx in range(min(seq_len, onset + 1), seq_len):
                    X[sample_idx, time_idx, neighbor_idx, [1, 2, 3]] += np.array([0.18, 0.22, 0.16], dtype=np.float32) * (
                        1.0 if root_hidden else 0.7
                    )

        metadata_rows.append(
            {
                "window_id": sample_idx,
                "fault_mode": fault_mode,
                "hidden_target": bool(primary_hidden),
                "primary_cause": CAUSE_NAMES[primary_cause],
                "primary_target": node_names[primary_target_idx],
            }
        )

    for hidden_idx in hidden_target_indices:
        X[:, :, hidden_idx, [0, 6, 7]] *= 0.15

    metadata = pd.DataFrame(metadata_rows)
    return SyntheticBundle(
        X=X,
        y_anomaly=y_anomaly,
        y_cause=y_cause,
        y_target=y_target,
        adjacency=adjacency,
        node_names=node_names,
        metadata=metadata,
    )


def split_indices(sample_count: int, seed: int = 42):
    indices = np.arange(sample_count)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_end = int(sample_count * 0.70)
    val_end = int(sample_count * 0.85)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def train_stgnn(
    bundle: SyntheticBundle,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hidden_dim: int = 48,
    epochs: int = 70,
):
    torch.manual_seed(42)
    model = SpatioTemporalGraphModel(
        input_dim=bundle.X.shape[-1],
        hidden_dim=hidden_dim,
        cause_classes=int(bundle.y_cause.max()) + 1,
        target_classes=int(bundle.y_target.max()) + 1,
        target_node_indices=list(range(bundle.X.shape[2])),
        use_topology=True,
        use_temporal=True,
        dropout=0.15,
    )

    x_train = torch.tensor(bundle.X[train_idx], dtype=torch.float32)
    x_val = torch.tensor(bundle.X[val_idx], dtype=torch.float32)
    adjacency_tensor = torch.tensor(bundle.adjacency, dtype=torch.float32)
    anomaly_train = torch.tensor(bundle.y_anomaly[train_idx], dtype=torch.float32)
    cause_train = torch.tensor(bundle.y_cause[train_idx], dtype=torch.long)
    target_train = torch.tensor(bundle.y_target[train_idx], dtype=torch.long)

    anomaly_counts = np.bincount(bundle.y_anomaly[train_idx], minlength=2)
    anomaly_loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(anomaly_counts[0] / max(anomaly_counts[1], 1), dtype=torch.float32)
    )

    cause_counts = np.bincount(bundle.y_cause[train_idx], minlength=int(bundle.y_cause.max()) + 1)
    target_counts = np.bincount(bundle.y_target[train_idx], minlength=int(bundle.y_target.max()) + 1)
    cause_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(cause_counts.sum() / np.clip(cause_counts, 1, None), dtype=torch.float32))
    target_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(target_counts.sum() / np.clip(target_counts, 1, None), dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    best_score = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train, adjacency_tensor)
        loss = (
            anomaly_loss(outputs["anomaly_logits"], anomaly_train)
            + cause_loss(outputs["cause_logits"], cause_train)
            + target_loss(outputs["target_logits"], target_train)
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation = model(x_val, adjacency_tensor)
            anomaly_pred = (torch.sigmoid(validation["anomaly_logits"]) > 0.5).int().cpu().numpy()
            cause_pred = torch.argmax(validation["cause_logits"], dim=1).cpu().numpy()
            target_pred = torch.argmax(validation["target_logits"], dim=1).cpu().numpy()

        score = (
            f1_score(bundle.y_anomaly[val_idx], anomaly_pred, zero_division=0)
            + f1_score(bundle.y_cause[val_idx], cause_pred, average="weighted", zero_division=0)
            + f1_score(bundle.y_target[val_idx], target_pred, average="weighted", zero_division=0)
        )
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def measure_rf_latency(model: RandomForestClassifier, test_matrix: np.ndarray, repeats: int = 30) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(test_matrix)
    elapsed = time.perf_counter() - start
    return elapsed / repeats / max(len(test_matrix), 1) * 1000.0


def measure_stgnn_latency(model: SpatioTemporalGraphModel, test_tensor: np.ndarray, adjacency: np.ndarray, repeats: int = 30) -> float:
    model.eval()
    x_tensor = torch.tensor(test_tensor, dtype=torch.float32)
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(repeats):
            model(x_tensor, adjacency_tensor)
        elapsed = time.perf_counter() - start
    return elapsed / repeats / max(len(test_tensor), 1) * 1000.0


def build_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    models = summary_df["Model"].tolist()
    x = np.arange(len(models))

    axes[0].bar(x - 0.16, summary_df["OverallTargetAccuracy"], width=0.32, color="#1d4ed8", label="Overall target")
    axes[0].bar(x + 0.16, summary_df["HiddenTargetAccuracy"], width=0.32, color="#0f766e", label="Hidden target")
    axes[0].set_xticks(x, models)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Target Localization")
    axes[0].legend(loc="lower left")

    axes[1].bar(x - 0.16, summary_df["SingleFaultCauseAccuracy"], width=0.32, color="#2563eb", label="Single fault")
    axes[1].bar(x + 0.16, summary_df["SimultaneousFaultCauseAccuracy"], width=0.32, color="#b45309", label="Simultaneous faults")
    axes[1].set_xticks(x, models)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Cause Accuracy")
    axes[1].set_title("Fault Complexity")
    axes[1].legend(loc="lower left")

    bars = axes[2].bar(x, summary_df["InferenceLatencyMs"], color=["#6b7280", "#f59e0b"])
    axes[2].set_xticks(x, models)
    axes[2].set_ylabel("Latency (ms/window)")
    axes[2].set_title("Inference Cost")
    twin = axes[2].twinx()
    twin.plot(x, summary_df["ThroughputWindowsPerSecond"], color="#0f766e", marker="o", linewidth=2)
    twin.set_ylabel("Throughput (windows/s)")

    for bar, latency in zip(bars, summary_df["InferenceLatencyMs"]):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03, f"{latency:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("59-Node Synthetic Clos Stress Study", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_why_graph_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    topology_cause = pd.read_csv(RESULTS_DIR / "topology_benchmark_cause.csv").set_index("Model")
    rf_row = summary_df.loc[summary_df["Model"] == "RandomForest"].iloc[0]
    stgnn_row = summary_df.loc[summary_df["Model"] == "STGNN-Full"].iloc[0]

    return pd.DataFrame(
        [
            {
                "Capability": "Public benchmark aggregate cause F1",
                "RandomForest": float(topology_cause.loc["RandomForest", "F1Weighted"]),
                "STGNN-Full": float(topology_cause.loc["STGNN-Full", "F1Weighted"]),
                "Units": "F1Weighted",
            },
            {
                "Capability": "59-node hidden-target accuracy",
                "RandomForest": float(rf_row["HiddenTargetAccuracy"]),
                "STGNN-Full": float(stgnn_row["HiddenTargetAccuracy"]),
                "Units": "Accuracy",
            },
            {
                "Capability": "59-node simultaneous-fault cause accuracy",
                "RandomForest": float(rf_row["SimultaneousFaultCauseAccuracy"]),
                "STGNN-Full": float(stgnn_row["SimultaneousFaultCauseAccuracy"]),
                "Units": "Accuracy",
            },
            {
                "Capability": "Propagation tracing",
                "RandomForest": "No",
                "STGNN-Full": "Yes",
                "Units": "qualitative",
            },
        ]
    )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    GRAPHS_DIR.mkdir(exist_ok=True)

    bundle = generate_scaleup_dataset()
    train_idx, val_idx, test_idx = split_indices(len(bundle.X))
    anomaly_train_mask = bundle.y_cause[train_idx] > 0
    anomaly_test_mask = bundle.y_cause[test_idx] > 0

    flattened = bundle.X.reshape(len(bundle.X), -1)

    rf_target = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    rf_target.fit(flattened[train_idx][anomaly_train_mask], bundle.y_target[train_idx][anomaly_train_mask])
    rf_target_pred = rf_target.predict(flattened[test_idx][anomaly_test_mask])

    rf_cause = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    rf_cause.fit(flattened[train_idx][anomaly_train_mask], bundle.y_cause[train_idx][anomaly_train_mask])
    rf_cause_pred = rf_cause.predict(flattened[test_idx][anomaly_test_mask])

    stgnn = train_stgnn(bundle, train_idx, val_idx)
    stgnn.eval()
    with torch.no_grad():
        outputs = stgnn(
            torch.tensor(bundle.X[test_idx], dtype=torch.float32),
            torch.tensor(bundle.adjacency, dtype=torch.float32),
        )
        stgnn_target_pred = torch.argmax(outputs["target_logits"], dim=1).cpu().numpy()[anomaly_test_mask]
        stgnn_cause_pred = torch.argmax(outputs["cause_logits"], dim=1).cpu().numpy()[anomaly_test_mask]

    test_metadata = bundle.metadata.iloc[test_idx].reset_index(drop=True)
    anomaly_metadata = test_metadata.loc[anomaly_test_mask].reset_index(drop=True)
    true_target = bundle.y_target[test_idx][anomaly_test_mask]
    true_cause = bundle.y_cause[test_idx][anomaly_test_mask]
    hidden_mask = anomaly_metadata["hidden_target"].to_numpy(dtype=bool)
    single_mask = anomaly_metadata["fault_mode"].to_numpy() == "single"
    simultaneous_mask = anomaly_metadata["fault_mode"].to_numpy() == "simultaneous"

    summary_rows = [
        {
            "Model": "RandomForest",
            "Topology": GRAPH_CONFIG["topology_name"],
            "NodeCount": len(bundle.node_names),
            "WindowCount": len(bundle.X),
            "AnomalousWindows": int(anomaly_test_mask.sum()),
            "HiddenTargetAccuracy": accuracy_score(true_target[hidden_mask], rf_target_pred[hidden_mask]),
            "OverallTargetAccuracy": accuracy_score(true_target, rf_target_pred),
            "SingleFaultCauseAccuracy": accuracy_score(true_cause[single_mask], rf_cause_pred[single_mask]),
            "SimultaneousFaultCauseAccuracy": accuracy_score(true_cause[simultaneous_mask], rf_cause_pred[simultaneous_mask]),
            "InferenceLatencyMs": measure_rf_latency(rf_target, flattened[test_idx][anomaly_test_mask]),
        },
        {
            "Model": "STGNN-Full",
            "Topology": GRAPH_CONFIG["topology_name"],
            "NodeCount": len(bundle.node_names),
            "WindowCount": len(bundle.X),
            "AnomalousWindows": int(anomaly_test_mask.sum()),
            "HiddenTargetAccuracy": accuracy_score(true_target[hidden_mask], stgnn_target_pred[hidden_mask]),
            "OverallTargetAccuracy": accuracy_score(true_target, stgnn_target_pred),
            "SingleFaultCauseAccuracy": accuracy_score(true_cause[single_mask], stgnn_cause_pred[single_mask]),
            "SimultaneousFaultCauseAccuracy": accuracy_score(true_cause[simultaneous_mask], stgnn_cause_pred[simultaneous_mask]),
            "InferenceLatencyMs": measure_stgnn_latency(stgnn, bundle.X[test_idx], bundle.adjacency),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df["ThroughputWindowsPerSecond"] = 1000.0 / summary_df["InferenceLatencyMs"]
    summary_df.to_csv(RESULTS_DIR / "synthetic_scaleup_summary.csv", index=False)

    why_graph_df = build_why_graph_table(summary_df)
    why_graph_df.to_csv(RESULTS_DIR / "why_graph_model.csv", index=False)

    build_figure(summary_df, GRAPHS_DIR / "synthetic_scaleup_performance.png")

    print(summary_df.to_string(index=False))
    print(why_graph_df.to_string(index=False))


if __name__ == "__main__":
    main()
