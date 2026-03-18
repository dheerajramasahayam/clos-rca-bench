from __future__ import annotations

import os
import time
import json
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from remediation_engine.digital_twin import load_canonical_topology


RAW_ROOT = Path("dataset/cisco_topology_benchmark/raw")
NODE_POSITIONS = {
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
DISRUPTIVE_EVENTS = {
    "enable_bfd",
    "add_blackhole",
    "set_loopback",
    "shutdown_interface",
    "clear_bgp",
    "add_network_loop",
    "remove_network_loop",
}


def load_feature_names(data_dir: Path) -> list[str]:
    with open(data_dir / "feature_names.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_scenario_catalog(scenario_names: list[str]) -> pd.DataFrame:
    rows = []
    for scenario_name in scenario_names:
        events_path = RAW_ROOT / scenario_name / "events.csv"
        if not events_path.exists():
            continue

        events = pd.read_csv(events_path)
        traffic = events.loc[events["event"] == "ixchariot_traffic", "timestamp"]
        stopped = events.loc[events["event"] == "ixchariot_traffic_stopped", "timestamp"]
        if traffic.empty or stopped.empty:
            continue

        traffic_start = float(traffic.iloc[0])
        traffic_stop = float(stopped.iloc[0])
        disruptive = events.loc[events["event"].isin(DISRUPTIVE_EVENTS)].copy()
        disruptive["timestamp"] = pd.to_numeric(disruptive["timestamp"], errors="coerce")
        disruptive = disruptive.dropna(subset=["timestamp"]).sort_values("timestamp")
        disruptive = disruptive.loc[disruptive["timestamp"] >= traffic_start]

        compound = (
            "evtmix" in scenario_name.lower()
            or "netloop" in scenario_name.lower()
            or disruptive["event"].nunique() > 1
            or len(disruptive) > 1
        )
        anomaly_start = float(disruptive.iloc[0]["timestamp"]) if not disruptive.empty else traffic_start
        primary_event = disruptive.iloc[0]["event"] if not disruptive.empty else "ixchariot_traffic"
        primary_device = disruptive.iloc[0].get("device", "") if not disruptive.empty else ""

        rows.append(
            {
                "scenario": scenario_name,
                "traffic_start": traffic_start,
                "traffic_stop": traffic_stop,
                "anomaly_start": anomaly_start,
                "primary_event": primary_event,
                "primary_device": primary_device,
                "disruptive_event_count": int(len(disruptive)),
                "scenario_type": "CompoundFailure" if compound else "SingleFailure",
            }
        )

    return pd.DataFrame(rows)


def _safe_cosine(matrix: np.ndarray, template: np.ndarray) -> np.ndarray:
    matrix_norm = np.linalg.norm(matrix, axis=1)
    template_norm = np.linalg.norm(template)
    denom = np.clip(matrix_norm * template_norm, 1e-8, None)
    return (matrix @ template) / denom


def _tune_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    candidate_thresholds = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 41)))
    best_threshold = float(np.median(scores))
    best_score = -1.0
    for threshold in candidate_thresholds:
        preds = (scores >= threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _measure_numpy_latency(fn, array: np.ndarray, repeats: int = 50) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn(array)
    elapsed = time.perf_counter() - start
    return elapsed / repeats / max(len(array), 1) * 1000.0


def measure_graph_latency(model, X_test: np.ndarray, adjacency: np.ndarray, repeats: int = 10) -> float:
    model.eval()
    x_tensor = torch.FloatTensor(X_test)
    adjacency_tensor = torch.FloatTensor(adjacency)
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(repeats):
            model(x_tensor, adjacency_tensor)
        elapsed = time.perf_counter() - start
    return elapsed / repeats / max(len(X_test), 1) * 1000.0


def evaluate_specialized_baselines(
    X: np.ndarray,
    y_anomaly: np.ndarray,
    y_cause: np.ndarray,
    y_target: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    node_names: list[str],
    label_maps: dict,
    feature_names: list[str],
):
    flat_X = X.reshape(X.shape[0], -1)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    node_index = {name: idx for idx, name in enumerate(node_names)}
    target_name_to_id = {
        value: int(key)
        for key, value in label_maps["target"].items()
    }

    anomaly_rows = []
    cause_rows = []
    target_rows = []
    prediction_bundle = {}
    latency_map = {}

    # Correlation-based RCA baseline
    train_flat = flat_X[train_idx]
    val_flat = flat_X[val_idx]
    test_flat = flat_X[test_idx]
    train_anomaly = y_anomaly[train_idx]
    val_anomaly = y_anomaly[val_idx]

    templates = {
        "normal": train_flat[train_anomaly == 0].mean(axis=0),
        "anomaly": train_flat[train_anomaly == 1].mean(axis=0),
        "cause": {
            label: train_flat[y_cause[train_idx] == label].mean(axis=0)
            for label in sorted(np.unique(y_cause[train_idx]))
            if label > 0
        },
        "target": {
            label: train_flat[y_target[train_idx] == label].mean(axis=0)
            for label in sorted(np.unique(y_target[train_idx]))
            if label > 0
        },
    }

    def predict_correlation(flat_input: np.ndarray):
        anomaly_scores = _safe_cosine(flat_input, templates["anomaly"]) - _safe_cosine(flat_input, templates["normal"])
        cause_labels = np.array(sorted(templates["cause"]))
        cause_matrix = np.column_stack([_safe_cosine(flat_input, templates["cause"][label]) for label in cause_labels])
        target_labels = np.array(sorted(templates["target"]))
        target_matrix = np.column_stack([_safe_cosine(flat_input, templates["target"][label]) for label in target_labels])
        return anomaly_scores, cause_labels[cause_matrix.argmax(axis=1)], target_labels[target_matrix.argmax(axis=1)]

    correlation_val_scores, _, _ = predict_correlation(val_flat)
    correlation_threshold = _tune_threshold(correlation_val_scores, val_anomaly)
    correlation_scores, correlation_cause_pred, correlation_target_pred = predict_correlation(test_flat)
    correlation_anomaly_pred = (correlation_scores >= correlation_threshold).astype(int)
    correlation_cause_pred = np.where(correlation_anomaly_pred == 1, correlation_cause_pred, 0)
    correlation_target_pred = np.where(correlation_anomaly_pred == 1, correlation_target_pred, 0)

    anomaly_rows.append(
        {
            "Model": "CorrelationRCA",
            "Accuracy": accuracy_score(y_anomaly[test_idx], correlation_anomaly_pred),
            "Precision": 0.0 if correlation_anomaly_pred.sum() == 0 else None,
        }
    )
    anomaly_rows[-1].update(_binary_metrics(y_anomaly[test_idx], correlation_anomaly_pred))
    anomaly_rows[-1]["Scores"] = correlation_scores
    anomaly_mask = y_cause[test_idx] > 0
    cause_rows.append({"Model": "CorrelationRCA", **_multiclass_metrics(y_cause[test_idx][anomaly_mask], correlation_cause_pred[anomaly_mask])})
    target_rows.append({"Model": "CorrelationRCA", **_multiclass_metrics(y_target[test_idx][anomaly_mask], correlation_target_pred[anomaly_mask])})
    prediction_bundle["CorrelationRCA"] = {
        "anomaly_pred": correlation_anomaly_pred,
        "cause_pred": correlation_cause_pred,
        "target_pred": correlation_target_pred,
        "anomaly_scores": correlation_scores,
    }
    latency_map["CorrelationRCA"] = _measure_numpy_latency(lambda arr: predict_correlation(arr), test_flat)

    # Rule-based RCA baseline
    idx = feature_index
    spine4_idx = node_index["spine4-3464"]
    spine3_idx = node_index["spine3"]
    leaf4_idx = node_index["leaf4"]
    leaf7_idx = node_index["leaf7"]

    def positive_mean(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0.0).mean(axis=1)

    def rule_scores(x_input: np.ndarray):
        bfd_score = positive_mean(x_input[:, :, spine4_idx, idx["bfd_down_count"]])
        blackhole_score = (
            positive_mean(x_input[:, :, :, idx["fib_total_drop_packets_delta"]].mean(axis=2))
            + positive_mean(x_input[:, :, :, idx["fib_unresolved_prefix_delta"]].mean(axis=2))
        )
        ecmp_score = (
            positive_mean(np.abs(x_input[:, :, spine3_idx, idx["bgp_path_count"]]))
            + positive_mean(np.abs(x_input[:, :, spine3_idx, idx["bgp_updates_received"]]))
            + positive_mean(np.abs(x_input[:, :, spine3_idx, idx["output_load_max"]]))
        )
        leaf4_score = (
            positive_mean(x_input[:, :, leaf4_idx, idx["interface_oper_down_count"]])
            + positive_mean(x_input[:, :, leaf4_idx, idx["carrier_transitions_delta"]])
            + positive_mean(x_input[:, :, leaf4_idx, idx["input_drops_delta"]])
        )
        leaf7_score = (
            positive_mean(x_input[:, :, leaf7_idx, idx["interface_oper_down_count"]])
            + positive_mean(x_input[:, :, leaf7_idx, idx["carrier_transitions_delta"]])
            + positive_mean(x_input[:, :, leaf7_idx, idx["input_drops_delta"]])
        )
        interface_score = np.maximum(leaf4_score, leaf7_score)
        score_matrix = np.column_stack([bfd_score, blackhole_score, ecmp_score, interface_score])
        return score_matrix, leaf4_score, leaf7_score

    rule_val_scores, _, _ = rule_scores(X[val_idx])
    rule_threshold = _tune_threshold(rule_val_scores.max(axis=1), val_anomaly)
    rule_score_matrix, rule_leaf4, rule_leaf7 = rule_scores(X[test_idx])
    rule_anomaly_scores = rule_score_matrix.max(axis=1)
    rule_anomaly_pred = (rule_anomaly_scores >= rule_threshold).astype(int)
    rule_cause_pred = rule_score_matrix.argmax(axis=1) + 1
    rule_cause_pred = np.where(rule_anomaly_pred == 1, rule_cause_pred, 0)

    rule_target_pred = np.zeros(len(test_idx), dtype=int)
    for index, cause_label in enumerate(rule_cause_pred):
        if cause_label == 1:
            rule_target_pred[index] = target_name_to_id["spine4-3464"]
        elif cause_label == 2:
            rule_target_pred[index] = target_name_to_id["leaf3"]
        elif cause_label == 3:
            rule_target_pred[index] = target_name_to_id["spine3"]
        elif cause_label == 4:
            rule_target_pred[index] = target_name_to_id["leaf4"] if rule_leaf4[index] >= rule_leaf7[index] else target_name_to_id["leaf7"]

    anomaly_rows.append({"Model": "RuleBasedRCA", **_binary_metrics(y_anomaly[test_idx], rule_anomaly_pred), "Scores": rule_anomaly_scores})
    cause_rows.append({"Model": "RuleBasedRCA", **_multiclass_metrics(y_cause[test_idx][anomaly_mask], rule_cause_pred[anomaly_mask])})
    target_rows.append({"Model": "RuleBasedRCA", **_multiclass_metrics(y_target[test_idx][anomaly_mask], rule_target_pred[anomaly_mask])})
    prediction_bundle["RuleBasedRCA"] = {
        "anomaly_pred": rule_anomaly_pred,
        "cause_pred": rule_cause_pred,
        "target_pred": rule_target_pred,
        "anomaly_scores": rule_anomaly_scores,
    }
    latency_map["RuleBasedRCA"] = _measure_numpy_latency(lambda arr: rule_scores(arr), X[test_idx])

    return anomaly_rows, cause_rows, target_rows, prediction_bundle, latency_map


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FalsePositiveRate": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "PrecisionWeighted": 0.0 if len(y_true) == 0 else float(
            pd.Series(y_pred).pipe(lambda _: __import__("sklearn.metrics").metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0))
        ),
        "RecallWeighted": 0.0 if len(y_true) == 0 else float(
            __import__("sklearn.metrics").metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "F1Weighted": 0.0 if len(y_true) == 0 else float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "F1Macro": 0.0 if len(y_true) == 0 else float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def build_temporal_tracking_outputs(
    test_metadata: pd.DataFrame,
    y_cause_test: np.ndarray,
    prediction_bundle: dict,
    cause_df: pd.DataFrame,
    latency_map: dict[str, float],
    scenario_catalog: pd.DataFrame,
):
    scenario_lookup = scenario_catalog.set_index("scenario")
    detail_rows = []
    summary_rows = []

    cause_accuracy_map = cause_df.set_index("Model")["Accuracy"].to_dict()

    for model_name, bundle in prediction_bundle.items():
        anomaly_pred = np.asarray(bundle["anomaly_pred"])
        model_rows = []
        for scenario_name, group in test_metadata.groupby("scenario", sort=False):
            if scenario_name not in scenario_lookup.index:
                continue
            info = scenario_lookup.loc[scenario_name]
            ordered = group.sort_values("window_start").copy()
            ordered_pred = anomaly_pred[ordered.index.to_numpy()]
            detected = int(ordered_pred.any())
            detection_delay = np.nan
            detection_window = np.nan
            if detected:
                first_hit = int(np.where(ordered_pred == 1)[0][0])
                detection_window = float(ordered.iloc[first_hit]["window_start"])
                detection_delay = max(0.0, detection_window - float(info["anomaly_start"]))

            row = {
                "Model": model_name,
                "scenario": scenario_name,
                "scenario_type": info["scenario_type"],
                "primary_event": info["primary_event"],
                "primary_device": info["primary_device"],
                "anomaly_start": info["anomaly_start"],
                "detected": detected,
                "detection_window_start": detection_window,
                "detection_delay_seconds": detection_delay,
            }
            detail_rows.append(row)
            model_rows.append(row)

        model_frame = pd.DataFrame(model_rows)
        summary_rows.append(
            {
                "Model": model_name,
                "RCAAccuracy": cause_accuracy_map.get(model_name, np.nan),
                "MeanDetectionDelaySeconds": model_frame["detection_delay_seconds"].dropna().mean(),
                "DetectionRecall": model_frame["detected"].mean() if not model_frame.empty else 0.0,
                "InferenceLatencyMs": latency_map.get(model_name, np.nan),
            }
        )

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def build_multi_failure_outputs(
    test_metadata: pd.DataFrame,
    y_cause_test: np.ndarray,
    y_target_test: np.ndarray,
    prediction_bundle: dict,
    scenario_catalog: pd.DataFrame,
):
    scenario_type_map = scenario_catalog.set_index("scenario")["scenario_type"].to_dict()
    anomaly_mask = y_cause_test > 0
    scenario_slice = test_metadata["scenario"].map(scenario_type_map).fillna("SingleFailure").to_numpy()

    rows = []
    for model_name, bundle in prediction_bundle.items():
        cause_pred = np.asarray(bundle["cause_pred"])
        target_pred = np.asarray(bundle["target_pred"])
        for slice_name in ["SingleFailure", "CompoundFailure"]:
            mask = anomaly_mask & (scenario_slice == slice_name)
            if not mask.any():
                continue
            rows.append(
                {
                    "Model": model_name,
                    "Slice": slice_name,
                    "CauseAccuracy": accuracy_score(y_cause_test[mask], cause_pred[mask]),
                    "TargetF1Weighted": f1_score(y_target_test[mask], target_pred[mask], average="weighted", zero_division=0),
                    "Windows": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def build_positioning_outputs(
    temporal_summary_df: pd.DataFrame,
    multi_failure_df: pd.DataFrame,
):
    family_map = {
        "RuleBasedRCA": ("Rule-based RCA", "RuleBasedRCA"),
        "CorrelationRCA": ("Correlation-based RCA", "CorrelationRCA"),
        "STGNN-NoTemporal": ("Graph-based RCA baseline", "GCN-Static"),
        "STGNN-Full": ("Temporal graph RCA", "STGNN-Full"),
    }
    compound_accuracy = (
        multi_failure_df.loc[multi_failure_df["Slice"] == "CompoundFailure", ["Model", "CauseAccuracy"]]
        .rename(columns={"CauseAccuracy": "CompoundFailureAccuracy"})
        .set_index("Model")
    )

    rows = []
    for model_name, (family, display_name) in family_map.items():
        if model_name not in set(temporal_summary_df["Model"]):
            continue
        row = temporal_summary_df.loc[temporal_summary_df["Model"] == model_name].iloc[0]
        rows.append(
            {
                "Method": display_name,
                "Family": family,
                "RCAAccuracy": row["RCAAccuracy"],
                "DetectionDelaySeconds": row["MeanDetectionDelaySeconds"],
                "InferenceLatencyMs": row["InferenceLatencyMs"],
                "CompoundFailureAccuracy": compound_accuracy.loc[model_name, "CompoundFailureAccuracy"] if model_name in compound_accuracy.index else np.nan,
                "SpeedTier": _speed_tier(row["InferenceLatencyMs"]),
            }
        )
    return pd.DataFrame(rows)


def _speed_tier(latency_ms: float) -> str:
    if np.isnan(latency_ms):
        return "unknown"
    if latency_ms < 0.5:
        return "fast"
    if latency_ms < 2.0:
        return "medium"
    return "higher"


def save_detection_delay_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    filtered = summary_df.loc[summary_df["Model"].isin(["RuleBasedRCA", "CorrelationRCA", "STGNN-NoTemporal", "STGNN-Full"])].copy()
    if filtered.empty:
        return

    x = np.arange(len(filtered))
    fig, ax1 = plt.subplots(figsize=(10.5, 5.4))
    bars = ax1.bar(x, filtered["MeanDetectionDelaySeconds"], color="#0f766e", alpha=0.85)
    ax1.set_ylabel("Mean Detection Delay (s)")
    ax1.set_xticks(x, filtered["Model"], rotation=20, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, filtered["RCAAccuracy"], color="#b45309", marker="o", linewidth=2)
    ax2.set_ylabel("RCA Accuracy")
    ax2.set_ylim(0, 1.05)

    for bar, delay in zip(bars, filtered["MeanDetectionDelaySeconds"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{delay:.1f}s", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Temporal Root Cause Tracking: Accuracy vs Detection Delay", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_multi_failure_figure(multi_failure_df: pd.DataFrame, output_path: Path) -> None:
    filtered = multi_failure_df.loc[multi_failure_df["Model"].isin(["RuleBasedRCA", "CorrelationRCA", "STGNN-NoTemporal", "STGNN-Full"])].copy()
    if filtered.empty:
        return

    pivot = filtered.pivot(index="Model", columns="Slice", values="CauseAccuracy").fillna(0.0)
    x = np.arange(len(pivot))
    width = 0.32

    plt.figure(figsize=(10.5, 5.4))
    plt.bar(x - width / 2, pivot.get("SingleFailure", pd.Series(0, index=pivot.index)), width=width, color="#1d4ed8", label="SingleFailure")
    plt.bar(x + width / 2, pivot.get("CompoundFailure", pd.Series(0, index=pivot.index)), width=width, color="#b45309", label="CompoundFailure")
    plt.xticks(x, pivot.index, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Cause Accuracy")
    plt.title("Compound-Failure Slice Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_deployment_figure(positioning_df: pd.DataFrame, output_path: Path) -> None:
    latency_lookup = positioning_df.set_index("Method")["InferenceLatencyMs"].to_dict() if not positioning_df.empty else {}
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, "Telemetry + IDS", "NetFlow, SNMP,\nlogs, anomaly flags", "#dbeafe"),
        (0.20, "Feature Layer", "Windowing, graph tensors,\nfeature scaling", "#dcfce7"),
        (0.37, "Temporal RCA", "STGNN-Full\n%.2f ms/window" % latency_lookup.get("STGNN-Full", float("nan")), "#fde68a"),
        (0.54, "Causal Chain", "Propagation tracing,\ndelay estimation", "#fef3c7"),
        (0.71, "Safety Gate", "Action validation\nand recovery checks", "#fecaca"),
        (0.88, "Operator View", "Alert, root cause,\naffected nodes", "#e9d5ff"),
    ]

    width = 0.12
    height = 0.38
    y = 0.31
    for x, title, subtitle, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.5,
            edgecolor="#0f172a",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height * 0.63, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + width / 2, y + height * 0.34, subtitle, ha="center", va="center", fontsize=9.2)

    for left, right in zip(boxes, boxes[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (left[0] + width, y + height / 2),
                (right[0], y + height / 2),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.6,
                color="#0f172a",
            )
        )

    ax.text(0.02, 0.90, "Datacenter RCA Deployment Pipeline", fontsize=18, fontweight="bold", color="#111827")
    ax.text(0.02, 0.84, "Operational flow from anomaly signal to temporal RCA, propagation tracing, and operator-safe remediation.", fontsize=10.5, color="#374151")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_case_study_outputs(
    X: np.ndarray,
    metadata: pd.DataFrame,
    model,
    adjacency: np.ndarray,
    node_names: list[str],
    label_maps: dict,
    scenario_catalog: pd.DataFrame,
):
    preferred_candidates = [
        "S-200202_2014_evtmix-1",
        "S-200206_1852_evtmix-1",
        "S-200206_1929_evtmix-1",
        "S-200202_2155_evtmix-1",
        "200121_0803_ecmp",
    ]
    available = set(metadata["scenario"])
    preferred = next((name for name in preferred_candidates if name in available), next(iter(available)))

    anomalous_candidates = metadata.loc[metadata["cause_name"] != "normal", "scenario"].unique().tolist()
    if preferred not in anomalous_candidates:
        preferred = next((name for name in preferred_candidates if name in anomalous_candidates), anomalous_candidates[0])

    case_mask = metadata["scenario"] == preferred
    case_X = X[case_mask]
    case_metadata = metadata.loc[case_mask].sort_values("window_start").reset_index(drop=True)
    scenario_info = scenario_catalog.set_index("scenario").loc[preferred]
    anomaly_start = float(scenario_info["anomaly_start"])

    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(case_X), torch.FloatTensor(adjacency))
        anomaly_scores = torch.sigmoid(outputs["anomaly_logits"]).cpu().numpy()
        cause_pred = torch.argmax(outputs["cause_logits"], dim=1).cpu().numpy()
        target_pred = torch.argmax(outputs["target_logits"], dim=1).cpu().numpy()

    target_map = {int(key): value for key, value in label_maps["target"].items()}
    cause_map = {int(key): value for key, value in label_maps["cause"].items()}

    case_metadata["anomaly_score"] = anomaly_scores
    case_metadata["predicted_cause"] = [cause_map[int(value)] for value in cause_pred]
    case_metadata["predicted_target"] = [target_map[int(value)] for value in target_pred]

    activity = np.mean(np.abs(case_X), axis=(1, 3))
    pre_fault_mask = case_metadata["window_end"] < anomaly_start
    baseline_slice = activity[pre_fault_mask.to_numpy()]
    if len(baseline_slice) < 2:
        baseline_slice = activity[: max(2, min(3, len(activity)))]
    baseline_mean = baseline_slice.mean(axis=0)
    baseline_std = np.clip(baseline_slice.std(axis=0), 1e-3, None)
    z_activity = (activity - baseline_mean) / baseline_std

    root_target = case_metadata.loc[case_metadata["cause_name"] != "normal", "target_device"].iloc[0]
    root_cause = case_metadata.loc[case_metadata["cause_name"] != "normal", "cause_name"].iloc[0]
    root_onset = anomaly_start

    node_rows = []
    window_starts = case_metadata["window_start"].to_numpy()
    for node_idx, node_name in enumerate(node_names):
        exceed = np.where((z_activity[:, node_idx] > 1.5) & (window_starts >= anomaly_start))[0]
        onset = float(case_metadata.loc[exceed[0], "window_start"]) if len(exceed) else np.nan
        node_rows.append(
            {
                "scenario": preferred,
                "node": node_name,
                "first_activation_window": onset,
                "activation_delay_seconds": onset - anomaly_start if np.isfinite(onset) else np.nan,
                "peak_z_score": float(z_activity[:, node_idx].max()),
            }
        )
    node_frame = pd.DataFrame(node_rows)

    graph, _ = load_canonical_topology(node_names)
    impacted = (
        node_frame.loc[(node_frame["node"] != root_target) & node_frame["first_activation_window"].notna()]
        .sort_values(["first_activation_window", "peak_z_score"], ascending=[True, False])
        .head(4)
    )
    path_rows = []
    for row in impacted.itertuples(index=False):
        try:
            path = nx.shortest_path(graph, root_target, row.node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = [root_target, row.node]
        path_rows.append(
            {
                "scenario": preferred,
                "root_target": root_target,
                "impacted_node": row.node,
                "propagation_path": " -> ".join(path),
                "activation_delay_seconds": row.activation_delay_seconds,
                "peak_z_score": row.peak_z_score,
            }
        )

    summary = pd.DataFrame(
        [
            {
                "scenario": preferred,
                "root_cause": root_cause,
                "root_target": root_target,
                "predicted_cause": case_metadata.loc[case_metadata["anomaly_score"].idxmax(), "predicted_cause"],
                "predicted_target": case_metadata.loc[case_metadata["anomaly_score"].idxmax(), "predicted_target"],
                "anomaly_start": anomaly_start,
                "first_detected_window": float(case_metadata.loc[case_metadata["anomaly_score"].idxmax(), "window_start"]),
                "detection_delay_seconds": max(
                    0.0,
                    float(case_metadata.loc[case_metadata["anomaly_score"].idxmax(), "window_start"]) - anomaly_start,
                ),
                "trace_note": "This trace was collected from a controlled internal lab environment simulating enterprise traffic.",
            }
        ]
    )

    return summary, pd.DataFrame(path_rows), case_metadata, node_frame


def save_case_study_figure(
    summary_df: pd.DataFrame,
    path_df: pd.DataFrame,
    case_metadata: pd.DataFrame,
    node_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    if summary_df.empty:
        return

    root_target = summary_df.iloc[0]["root_target"]
    anomaly_start = summary_df.iloc[0]["anomaly_start"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.1, 1.0]})

    rel_time = case_metadata["window_start"] - anomaly_start
    ax1.plot(rel_time, case_metadata["anomaly_score"], color="#0f766e", linewidth=2, label="STGNN anomaly score")
    ax1.axvline(0, color="#b91c1c", linestyle="--", linewidth=1.5, label="Fault onset")
    ax1.set_xlabel("Seconds from fault onset")
    ax1.set_ylabel("Anomaly score")
    ax1.set_title("Case Study Timeline")
    ax1.legend(loc="upper left")

    ax2.axis("off")
    for source, target in zip(path_df["root_target"], path_df["impacted_node"]):
        if source not in NODE_POSITIONS or target not in NODE_POSITIONS:
            continue
        x1, y1 = NODE_POSITIONS[source]
        x2, y2 = NODE_POSITIONS[target]
        ax2.plot([x1, x2], [y1, y2], color="#cbd5e1", linewidth=1.5, zorder=1)

    delay_lookup = node_frame.set_index("node")["activation_delay_seconds"].to_dict()
    for node_name, (x, y) in NODE_POSITIONS.items():
        delay = delay_lookup.get(node_name, np.nan)
        if node_name == root_target:
            color = "#b91c1c"
        elif np.isfinite(delay):
            color = "#0f766e"
        else:
            color = "#94a3b8"
        ax2.scatter([x], [y], s=420, color=color, edgecolor="white", linewidth=1.5, zorder=2)
        label = node_name if not np.isfinite(delay) else f"{node_name}\n+{delay:.0f}s"
        ax2.text(x, y - 0.18, label, ha="center", va="top", fontsize=8.5)

    for row in path_df.itertuples(index=False):
        nodes = row.propagation_path.split(" -> ")
        for left, right in zip(nodes, nodes[1:]):
            if left not in NODE_POSITIONS or right not in NODE_POSITIONS:
                continue
            ax2.add_patch(
                FancyArrowPatch(
                    NODE_POSITIONS[left],
                    NODE_POSITIONS[right],
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=1.8,
                    color="#b45309",
                )
            )

    ax2.set_title("Propagation Paths")
    fig.suptitle("Real Failure Case Study: Temporal Detection and Propagation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
