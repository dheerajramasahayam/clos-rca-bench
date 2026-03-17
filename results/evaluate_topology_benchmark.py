import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from remediation_engine.safety_validator import recommend_action, validate_action
from remediation_engine.digital_twin import (
    evaluate_recovery,
    load_canonical_topology,
)
from root_cause_analysis.topology_rca_model import SpatioTemporalGraphModel, load_topology_benchmark
from telemetry_parser.topology_benchmark import preprocess_topology_benchmark


RESULTS_DIR = Path("results")
GRAPHS_DIR = Path("graphs")
DATA_DIR = Path("dataset/cisco_topology_benchmark/processed")


def ensure_benchmark():
    required = [
        DATA_DIR / "X_topology.npy",
        DATA_DIR / "adjacency.npy",
        DATA_DIR / "window_metadata.csv",
    ]
    if not all(path.exists() for path in required):
        preprocess_topology_benchmark()


def load_metadata():
    with open(DATA_DIR / "node_names.json", "r", encoding="utf-8") as handle:
        node_names = json.load(handle)
    with open(DATA_DIR / "label_maps.json", "r", encoding="utf-8") as handle:
        label_maps = json.load(handle)
    metadata = pd.read_csv(DATA_DIR / "window_metadata.csv")
    return node_names, label_maps, metadata


def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FalsePositiveRate": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def multiclass_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "PrecisionWeighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "RecallWeighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1Weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, labels, output_name, title):
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / output_name, dpi=200)
    plt.close()


def save_roc(y_true, scores, output_name, title):
    fpr, tpr, _ = roc_curve(y_true, scores)
    score_auc = auc(fpr, tpr)
    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr, tpr, linewidth=2, color="#0f766e", label=f"AUC = {score_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / output_name, dpi=200)
    plt.close()


def save_model_comparison(anomaly_df, cause_df, target_df):
    model_names = anomaly_df["Model"].tolist()
    x = np.arange(len(model_names))
    width = 0.25

    plt.figure(figsize=(11, 5.5))
    plt.bar(x - width, anomaly_df["F1"], width=width, label="Anomaly F1", color="#0f766e")
    plt.bar(x, cause_df["F1Weighted"], width=width, label="Cause F1", color="#b45309")
    plt.bar(x + width, target_df["F1Weighted"], width=width, label="Target F1", color="#1d4ed8")
    plt.xticks(x, model_names, rotation=20, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Topology Benchmark Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "topology_model_comparison.png", dpi=200)
    plt.close()


def save_topology_graph(adjacency, node_names):
    positions = {
        "spine1": (0.0, 2.0),
        "spine2": (1.0, 2.0),
        "spine3": (2.0, 2.0),
        "spine4-3464": (3.0, 2.0),
        "leaf3": (0.0, 1.0),
        "leaf4": (1.0, 1.0),
        "leaf5": (2.0, 1.0),
        "leaf7": (3.0, 1.0),
        "leaf8": (4.0, 1.0),
        "dr02": (1.5, 0.0),
        "dr03": (2.5, 0.0),
    }
    colors = {
        "spine": "#0f766e",
        "leaf": "#b45309",
        "dr": "#1d4ed8",
    }

    plt.figure(figsize=(7.8, 4.5))
    for src_idx, src_name in enumerate(node_names):
        for dst_idx in range(src_idx + 1, len(node_names)):
            if adjacency[src_idx, dst_idx] <= 0 or src_name not in positions or node_names[dst_idx] not in positions:
                continue
            x1, y1 = positions[src_name]
            x2, y2 = positions[node_names[dst_idx]]
            plt.plot([x1, x2], [y1, y2], color="#94a3b8", linewidth=1.2, zorder=1)

    for node_name, (x, y) in positions.items():
        if node_name not in node_names:
            continue
        if node_name.startswith("spine"):
            color = colors["spine"]
        elif node_name.startswith("leaf"):
            color = colors["leaf"]
        else:
            color = colors["dr"]
        plt.scatter([x], [y], s=420, color=color, edgecolor="white", linewidth=1.5, zorder=2)
        plt.text(x, y - 0.16, node_name, ha="center", va="top", fontsize=9)

    plt.axis("off")
    plt.title("Cisco Clos-Topology Benchmark Graph")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "cisco_clos_topology.png", dpi=200)
    plt.close()


def flatten_windows(X):
    return X.reshape(X.shape[0], -1)


def evaluate_tabular_baselines(X_train, X_test, y_train, y_test, task_name):
    models = [
        ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
        ("MLP", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)),
    ]

    rows = []
    predictions = {}

    for model_name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[model_name] = preds

        if task_name == "anomaly":
            probas = model.predict_proba(X_test)[:, 1]
            metrics = binary_metrics(y_test, preds)
            rows.append({"Model": model_name, **metrics, "Scores": probas})
        else:
            metrics = multiclass_metrics(y_test, preds)
            rows.append({"Model": model_name, **metrics})

    return rows, predictions


def train_graph_model(
    X,
    y_anomaly,
    y_cause,
    y_target,
    adjacency,
    train_idx,
    val_idx,
    target_node_indices,
    use_topology=True,
    use_temporal=True,
):
    torch.manual_seed(42)

    model = SpatioTemporalGraphModel(
        input_dim=X.shape[-1],
        hidden_dim=64,
        cause_classes=int(y_cause.max()) + 1,
        target_classes=int(y_target.max()) + 1,
        target_node_indices=target_node_indices,
        use_topology=use_topology,
        use_temporal=use_temporal,
    )

    x_train = torch.FloatTensor(X[train_idx])
    x_val = torch.FloatTensor(X[val_idx])
    anomaly_train = torch.FloatTensor(y_anomaly[train_idx])
    anomaly_val = torch.FloatTensor(y_anomaly[val_idx])
    cause_train = torch.LongTensor(y_cause[train_idx])
    cause_val = torch.LongTensor(y_cause[val_idx])
    target_train = torch.LongTensor(y_target[train_idx])
    target_val = torch.LongTensor(y_target[val_idx])
    adjacency_tensor = torch.FloatTensor(adjacency)

    anomaly_counts = np.bincount(y_anomaly[train_idx], minlength=2)
    pos_weight = torch.tensor(
        anomaly_counts[0] / max(anomaly_counts[1], 1),
        dtype=torch.float32,
    )

    cause_counts = np.bincount(y_cause[train_idx], minlength=int(y_cause.max()) + 1)
    cause_weights = torch.FloatTensor(np.sum(cause_counts) / np.clip(cause_counts, 1, None))

    target_counts = np.bincount(y_target[train_idx], minlength=int(y_target.max()) + 1)
    target_weights = torch.FloatTensor(np.sum(target_counts) / np.clip(target_counts, 1, None))

    anomaly_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    cause_loss = nn.CrossEntropyLoss(weight=cause_weights)
    target_loss = nn.CrossEntropyLoss(weight=target_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_state = None
    best_score = -1.0

    for epoch in range(120):
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
            val_outputs = model(x_val, adjacency_tensor)
            val_anomaly = (torch.sigmoid(val_outputs["anomaly_logits"]) > 0.5).int().cpu().numpy()
            val_cause = torch.argmax(val_outputs["cause_logits"], dim=1).cpu().numpy()
            val_target = torch.argmax(val_outputs["target_logits"], dim=1).cpu().numpy()

        score = (
            f1_score(anomaly_val.numpy(), val_anomaly, zero_division=0)
            + f1_score(cause_val.numpy(), val_cause, average="weighted", zero_division=0)
            + f1_score(target_val.numpy(), val_target, average="weighted", zero_division=0)
        )
        if score > best_score:
            best_score = score
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def evaluate_graph_model(model_name, model, X_test, y_anomaly_test, y_cause_test, y_target_test, adjacency):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_test), torch.FloatTensor(adjacency))
        anomaly_scores = torch.sigmoid(outputs["anomaly_logits"]).cpu().numpy()
        anomaly_preds = (anomaly_scores > 0.5).astype(int)
        cause_preds = torch.argmax(outputs["cause_logits"], dim=1).cpu().numpy()
        target_preds = torch.argmax(outputs["target_logits"], dim=1).cpu().numpy()

    anomaly_mask = y_cause_test > 0
    anomaly_row = {"Model": model_name, **binary_metrics(y_anomaly_test, anomaly_preds), "Scores": anomaly_scores}
    cause_row = {
        "Model": model_name,
        **multiclass_metrics(y_cause_test[anomaly_mask], cause_preds[anomaly_mask]),
    }
    target_row = {
        "Model": model_name,
        **multiclass_metrics(y_target_test[anomaly_mask], target_preds[anomaly_mask]),
    }

    return anomaly_row, cause_row, target_row, anomaly_preds, cause_preds, target_preds, anomaly_scores


def evaluate_remediation(cause_preds, target_preds, y_cause_test, y_target_test, test_metadata, adjacency, node_names, label_maps):
    anomaly_mask = y_cause_test > 0
    cause_map = {int(key): value for key, value in label_maps["cause"].items()}
    target_map = {int(key): value for key, value in label_maps["target"].items()}

    rows = []
    for idx, is_anomaly in enumerate(anomaly_mask):
        if not is_anomaly:
            continue

        predicted_cause = cause_map[int(cause_preds[idx])]
        predicted_target = target_map[int(target_preds[idx])]
        ground_cause = cause_map[int(y_cause_test[idx])]
        ground_target = target_map[int(y_target_test[idx])]

        action = recommend_action(predicted_cause, predicted_target)
        safe, reason = validate_action(action, predicted_target, adjacency, node_names)
        ground_action = recommend_action(ground_cause, ground_target)

        rows.append(
            {
                "scenario": test_metadata.iloc[idx]["scenario"],
                "target_interface": test_metadata.iloc[idx]["target_interface"],
                "predicted_cause": predicted_cause,
                "predicted_target": predicted_target,
                "ground_cause": ground_cause,
                "ground_target": ground_target,
                "recommended_action": action["action_id"],
                "ground_action": ground_action["action_id"],
                "safe": int(safe),
                "action_match": int(
                    action["action_id"] == ground_action["action_id"] and predicted_target == ground_target
                ),
                "gate_reason": reason,
            }
        )

    detail_frame = pd.DataFrame(rows)
    metrics = {
        "ActionMatchRate": detail_frame["action_match"].mean() if not detail_frame.empty else 0.0,
        "SafetyPassRate": detail_frame["safe"].mean() if not detail_frame.empty else 0.0,
        "UnsafeBlockedRate": (
            1.0 - detail_frame.loc[detail_frame["action_match"] == 0, "safe"].mean()
            if (detail_frame["action_match"] == 0).any()
            else 0.0
        ),
        "RecoveryEligibleRate": (
            (detail_frame["action_match"] & detail_frame["safe"]).mean() if not detail_frame.empty else 0.0
        ),
        "TestWindows": len(detail_frame),
    }
    return detail_frame, metrics


def evaluate_digital_twin(remediation_details, node_names):
    graph, interface_map = load_canonical_topology(node_names)

    rows = []
    for row in remediation_details.itertuples(index=False):
        metrics = evaluate_recovery(
            graph=graph,
            interface_map=interface_map,
            cause_name=row.ground_cause,
            ground_target=row.ground_target,
            target_interface=row.target_interface,
            action_id=row.recommended_action,
            predicted_target=row.predicted_target,
            safe=bool(row.safe),
        )
        rows.append(
            {
                "scenario": row.scenario,
                "ground_cause": row.ground_cause,
                "ground_target": row.ground_target,
                "predicted_target": row.predicted_target,
                "recommended_action": row.recommended_action,
                "safe": row.safe,
                **metrics,
            }
        )

    detail_frame = pd.DataFrame(rows)
    summary = {
        "RecoverySuccessRate": detail_frame["RecoverySuccess"].mean() if not detail_frame.empty else 0.0,
        "MeanReachabilityGain": detail_frame["ReachabilityGain"].mean() if not detail_frame.empty else 0.0,
        "MeanBlastRadiusReduction": detail_frame["BlastRadiusReduction"].mean() if not detail_frame.empty else 0.0,
        "MeanOverloadReduction": detail_frame["OverloadReduction"].mean() if not detail_frame.empty else 0.0,
        "RecoveredReachability": detail_frame["RecoveredReachability"].mean() if not detail_frame.empty else 0.0,
        "FaultReachability": detail_frame["FaultReachability"].mean() if not detail_frame.empty else 0.0,
        "TestWindows": len(detail_frame),
    }
    return detail_frame, summary


def save_recovery_figure(digital_twin_details):
    grouped = (
        digital_twin_details.groupby("ground_cause")[
            ["FaultReachability", "RecoveredReachability", "FaultBlastRadius", "RecoveredBlastRadius"]
        ]
        .mean()
        .reset_index()
    )

    x = np.arange(len(grouped))
    width = 0.18

    plt.figure(figsize=(11, 5.5))
    plt.bar(x - 1.5 * width, grouped["FaultReachability"], width=width, color="#b91c1c", label="Fault Reachability")
    plt.bar(x - 0.5 * width, grouped["RecoveredReachability"], width=width, color="#15803d", label="Recovered Reachability")
    plt.bar(x + 0.5 * width, grouped["FaultBlastRadius"], width=width, color="#f59e0b", label="Fault Blast Radius")
    plt.bar(x + 1.5 * width, grouped["RecoveredBlastRadius"], width=width, color="#1d4ed8", label="Recovered Blast Radius")
    plt.xticks(x, grouped["ground_cause"], rotation=20, ha="right")
    plt.ylabel("Average score")
    plt.ylim(0, 1.05)
    plt.title("Counterfactual Recovery in the Topology Digital Twin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "topology_digital_twin_recovery.png", dpi=200)
    plt.close()


def evaluate_target_slices(target_predictions, y_target_test, target_labels):
    hidden_labels = {
        target_labels.index("leaf3"),
        target_labels.index("spine4-3464"),
    }
    observed_mask = (y_target_test > 0) & ~np.isin(y_target_test, list(hidden_labels))
    hidden_mask = np.isin(y_target_test, list(hidden_labels))

    rows = []
    for model_name, preds in target_predictions.items():
        observed_metrics = multiclass_metrics(y_target_test[observed_mask], preds[observed_mask]) if observed_mask.any() else {
            "Accuracy": 0.0,
            "PrecisionWeighted": 0.0,
            "RecallWeighted": 0.0,
            "F1Weighted": 0.0,
            "F1Macro": 0.0,
        }
        hidden_metrics = multiclass_metrics(y_target_test[hidden_mask], preds[hidden_mask]) if hidden_mask.any() else {
            "Accuracy": 0.0,
            "PrecisionWeighted": 0.0,
            "RecallWeighted": 0.0,
            "F1Weighted": 0.0,
            "F1Macro": 0.0,
        }
        rows.append(
            {
                "Model": model_name,
                "Slice": "ObservedTargets",
                **observed_metrics,
            }
        )
        rows.append(
            {
                "Model": model_name,
                "Slice": "HiddenTargets",
                **hidden_metrics,
            }
        )
    return pd.DataFrame(rows)


def main():
    ensure_benchmark()
    RESULTS_DIR.mkdir(exist_ok=True)
    GRAPHS_DIR.mkdir(exist_ok=True)

    X, y_anomaly, y_cause, y_target, adjacency, train_idx, val_idx, test_idx = load_topology_benchmark()
    node_names, label_maps, metadata = load_metadata()
    target_device_names = [label_maps["target"][str(idx)] for idx in range(1, len(label_maps["target"]))]
    target_node_indices = [node_names.index(device_name) for device_name in target_device_names]
    cause_labels = [label_maps["cause"][str(idx)] for idx in range(len(label_maps["cause"]))]
    target_labels = [label_maps["target"][str(idx)] for idx in range(len(label_maps["target"]))]

    flat_X = flatten_windows(X)
    anomaly_rows, anomaly_preds = evaluate_tabular_baselines(
        flat_X[train_idx],
        flat_X[test_idx],
        y_anomaly[train_idx],
        y_anomaly[test_idx],
        task_name="anomaly",
    )

    train_anomaly_mask = y_cause[train_idx] > 0
    test_anomaly_mask = y_cause[test_idx] > 0
    cause_rows, _ = evaluate_tabular_baselines(
        flat_X[train_idx][train_anomaly_mask],
        flat_X[test_idx][test_anomaly_mask],
        y_cause[train_idx][train_anomaly_mask],
        y_cause[test_idx][test_anomaly_mask],
        task_name="cause",
    )

    target_rows, target_baseline_predictions = evaluate_tabular_baselines(
        flat_X[train_idx][train_anomaly_mask],
        flat_X[test_idx][test_anomaly_mask],
        y_target[train_idx][train_anomaly_mask],
        y_target[test_idx][test_anomaly_mask],
        task_name="target",
    )

    graph_variants = [
        ("STGNN-Full", True, True),
        ("STGNN-NoTopology", False, True),
        ("STGNN-NoTemporal", True, False),
    ]

    graph_predictions = {}
    for model_name, use_topology, use_temporal in graph_variants:
        model = train_graph_model(
            X,
            y_anomaly,
            y_cause,
            y_target,
            adjacency,
            train_idx,
            val_idx,
            target_node_indices=target_node_indices,
            use_topology=use_topology,
            use_temporal=use_temporal,
        )
        (
            anomaly_row,
            cause_row,
            target_row,
            anomaly_pred,
            cause_pred,
            target_pred,
            anomaly_scores,
        ) = evaluate_graph_model(
            model_name,
            model,
            X[test_idx],
            y_anomaly[test_idx],
            y_cause[test_idx],
            y_target[test_idx],
            adjacency,
        )
        anomaly_rows.append(anomaly_row)
        cause_rows.append(cause_row)
        target_rows.append(target_row)
        graph_predictions[model_name] = {
            "model": model,
            "anomaly_pred": anomaly_pred,
            "cause_pred": cause_pred,
            "target_pred": target_pred,
            "anomaly_scores": anomaly_scores,
        }

    anomaly_df = pd.DataFrame([{key: value for key, value in row.items() if key != "Scores"} for row in anomaly_rows])
    cause_df = pd.DataFrame(cause_rows)
    target_df = pd.DataFrame(target_rows)

    anomaly_df.to_csv(RESULTS_DIR / "topology_benchmark_anomaly.csv", index=False)
    cause_df.to_csv(RESULTS_DIR / "topology_benchmark_cause.csv", index=False)
    target_df.to_csv(RESULTS_DIR / "topology_benchmark_target.csv", index=False)

    target_prediction_slices = {
        "RandomForest": target_baseline_predictions["RandomForest"],
        "MLP": target_baseline_predictions["MLP"],
        "STGNN-Full": graph_predictions["STGNN-Full"]["target_pred"][test_anomaly_mask],
        "STGNN-NoTopology": graph_predictions["STGNN-NoTopology"]["target_pred"][test_anomaly_mask],
        "STGNN-NoTemporal": graph_predictions["STGNN-NoTemporal"]["target_pred"][test_anomaly_mask],
    }
    target_slice_df = evaluate_target_slices(
        target_prediction_slices,
        y_target[test_idx][test_anomaly_mask],
        target_labels,
    )
    target_slice_df.to_csv(RESULTS_DIR / "topology_benchmark_target_slices.csv", index=False)

    rf_scores = next(row["Scores"] for row in anomaly_rows if row["Model"] == "RandomForest")
    save_roc(y_anomaly[test_idx], rf_scores, "topology_rf_roc.png", "Topology Benchmark ROC - RandomForest")

    full_scores = graph_predictions["STGNN-Full"]["anomaly_scores"]
    save_roc(y_anomaly[test_idx], full_scores, "topology_stgnn_roc.png", "Topology Benchmark ROC - STGNN")

    test_anomaly_mask = y_cause[test_idx] > 0
    save_confusion_matrix(
        y_cause[test_idx][test_anomaly_mask],
        graph_predictions["STGNN-Full"]["cause_pred"][test_anomaly_mask],
        cause_labels,
        "topology_cause_cm.png",
        "Topology Benchmark Cause Confusion Matrix",
    )
    save_confusion_matrix(
        y_target[test_idx][test_anomaly_mask],
        graph_predictions["STGNN-Full"]["target_pred"][test_anomaly_mask],
        target_labels,
        "topology_target_cm.png",
        "Topology Benchmark Target Confusion Matrix",
    )

    test_metadata = metadata.iloc[test_idx].reset_index(drop=True)
    remediation_details, remediation_metrics = evaluate_remediation(
        graph_predictions["STGNN-Full"]["cause_pred"],
        graph_predictions["STGNN-Full"]["target_pred"],
        y_cause[test_idx],
        y_target[test_idx],
        test_metadata,
        adjacency,
        node_names,
        label_maps,
    )
    remediation_details.to_csv(RESULTS_DIR / "topology_benchmark_remediation_details.csv", index=False)
    pd.DataFrame([remediation_metrics]).to_csv(RESULTS_DIR / "topology_benchmark_remediation.csv", index=False)

    digital_twin_details, digital_twin_summary = evaluate_digital_twin(remediation_details, node_names)
    digital_twin_details.to_csv(RESULTS_DIR / "topology_benchmark_digital_twin_details.csv", index=False)
    pd.DataFrame([digital_twin_summary]).to_csv(RESULTS_DIR / "topology_benchmark_digital_twin.csv", index=False)

    leaderboard_rows = []
    for row in anomaly_df.itertuples(index=False):
        leaderboard_rows.append(
            {"Benchmark": "ClosRCA-Bench", "Task": "AnomalyDetection", "Model": row.Model, "PrimaryMetric": "F1", "Score": row.F1}
        )
    for row in cause_df.itertuples(index=False):
        leaderboard_rows.append(
            {"Benchmark": "ClosRCA-Bench", "Task": "CauseClassification", "Model": row.Model, "PrimaryMetric": "F1Weighted", "Score": row.F1Weighted}
        )
    for row in target_df.itertuples(index=False):
        leaderboard_rows.append(
            {"Benchmark": "ClosRCA-Bench", "Task": "TargetLocalization", "Model": row.Model, "PrimaryMetric": "F1Weighted", "Score": row.F1Weighted}
        )
    leaderboard_rows.append(
        {
            "Benchmark": "ClosRCA-Bench",
            "Task": "CounterfactualRecovery",
            "Model": "STGNN-Full+SafetyGate",
            "PrimaryMetric": "RecoverySuccessRate",
            "Score": digital_twin_summary["RecoverySuccessRate"],
        }
    )
    pd.DataFrame(leaderboard_rows).to_csv(RESULTS_DIR / "closrca_bench_leaderboard.csv", index=False)

    save_model_comparison(anomaly_df, cause_df, target_df)
    save_topology_graph(adjacency, node_names)
    save_recovery_figure(digital_twin_details)

    full_model = graph_predictions["STGNN-Full"]["model"]
    torch.save(
        {
            "model_state_dict": full_model.state_dict(),
            "input_dim": X.shape[-1],
            "hidden_dim": 64,
            "cause_classes": int(y_cause.max()) + 1,
            "target_classes": int(y_target.max()) + 1,
            "target_node_indices": target_node_indices,
            "node_names": node_names,
            "label_maps": label_maps,
        },
        RESULTS_DIR / "topology_benchmark_stgnn.pth",
    )

    print(anomaly_df.to_string(index=False))
    print(cause_df.to_string(index=False))
    print(target_df.to_string(index=False))
    print(target_slice_df.to_string(index=False))
    print(pd.DataFrame([remediation_metrics]).to_string(index=False))
    print(pd.DataFrame([digital_twin_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
