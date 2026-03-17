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
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from anomaly_detection_model.lstm_model import LSTMAnomalyDetector
from anomaly_detection_model.transformer_model import TransformerAnomalyDetector
from root_cause_analysis.rca_model import (
    TemporalGCNClassifier,
    build_temporal_adjacency,
    split_rca_dataset,
)

RESULTS_DIR = Path("results")
GRAPHS_DIR = Path("graphs")


def ensure_output_dirs():
    RESULTS_DIR.mkdir(exist_ok=True)
    GRAPHS_DIR.mkdir(exist_ok=True)


def compute_binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr_value = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FalsePositiveRate": fpr_value,
    }


def save_roc_curve(y_true, scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#d94801", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="#6b7280", lw=1.5, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / f"roc_{model_name.lower()}.png", dpi=200)
    plt.close()


def save_confusion_matrix(y_true, y_pred, model_name):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / f"cm_{model_name.lower()}.png", dpi=200)
    plt.close()


def save_anomaly_timeline():
    telemetry_path = Path("dataset/network_telemetry.csv")
    if not telemetry_path.exists():
        return

    df = pd.read_csv(telemetry_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").head(500)

    anomaly_points = df[df["is_anomaly"] == 1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(df["timestamp"], df["latency"], color="#0f766e", linewidth=1.6)
    axes[0].scatter(
        anomaly_points["timestamp"],
        anomaly_points["latency"],
        color="#b91c1c",
        s=18,
        label="Anomaly window",
    )
    axes[0].set_ylabel("Latency")
    axes[0].set_title("Synthetic Telemetry Timeline With Injected Anomalies")
    axes[0].legend(loc="upper right")

    axes[1].plot(
        df["timestamp"], df["packet_loss"], color="#1d4ed8", linewidth=1.6
    )
    axes[1].scatter(
        anomaly_points["timestamp"],
        anomaly_points["packet_loss"],
        color="#b91c1c",
        s=18,
    )
    axes[1].set_ylabel("Packet Loss")
    axes[1].set_xlabel("Timestamp")

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "anomaly_detection_timeline.png", dpi=200)
    plt.close()


def load_synthetic_test_split():
    X = np.load("dataset/X.npy")
    y = np.load("dataset/y_anomaly.npy")
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return torch.FloatTensor(X_test), y_test


def evaluate_synthetic_models():
    X_test, y_test = load_synthetic_test_split()
    input_dim = X_test.shape[2]

    models = [
        (
            "LSTM",
            LSTMAnomalyDetector(input_dim, 64, 2, 1),
            RESULTS_DIR / "lstm_model.pth",
        ),
        (
            "Transformer",
            TransformerAnomalyDetector(input_dim, 32, 4, 2),
            RESULTS_DIR / "transformer_model.pth",
        ),
    ]

    rows = []
    for model_name, model, checkpoint_path in models:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()

        with torch.no_grad():
            scores = model(X_test).cpu().numpy().flatten()
            predictions = (scores > 0.5).astype(int)

        metrics = compute_binary_metrics(y_test, predictions)
        rows.append({"Dataset": "Synthetic", "Model": model_name, **metrics})
        save_roc_curve(y_test, scores, model_name)
        save_confusion_matrix(y_test, predictions, model_name)

    return rows


def evaluate_real_model():
    real_inputs = Path("dataset/real_processed/X_gct.npy")
    real_labels = Path("dataset/real_processed/y_gct.npy")
    real_checkpoint = RESULTS_DIR / "real" / "lstm_real_model.pth"

    if not (real_inputs.exists() and real_labels.exists() and real_checkpoint.exists()):
        return []

    X = np.load(real_inputs)
    y = np.load(real_labels)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_test = torch.FloatTensor(X_test)
    model = LSTMAnomalyDetector(X_test.shape[2], 64, 2, 1)
    model.load_state_dict(torch.load(real_checkpoint, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        scores = model(X_test).cpu().numpy().flatten()
        predictions = (scores > 0.5).astype(int)

    metrics = compute_binary_metrics(y_test, predictions)
    return [{"Dataset": "GoogleClusterTrace", "Model": "LSTM", **metrics}]


def evaluate_rca_model():
    checkpoint_path = RESULTS_DIR / "rca_model.pth"
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _, X_test, _, y_test = split_rca_dataset()

    X_test = torch.FloatTensor(X_test)
    y_test = np.asarray(y_test)
    adjacency = build_temporal_adjacency(checkpoint["seq_len"])
    model = TemporalGCNClassifier(
        checkpoint["input_dim"],
        checkpoint["hidden_dim"],
        checkpoint["output_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(X_test, adjacency)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "PrecisionWeighted": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "RecallWeighted": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1Weighted": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
    }

    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "rca_metrics.csv", index=False)
    return metrics


def run_evaluation():
    ensure_output_dirs()

    rows = []
    rows.extend(evaluate_synthetic_models())
    rows.extend(evaluate_real_model())
    save_anomaly_timeline()

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    rca_metrics = evaluate_rca_model()

    print("Saved anomaly detection metrics to results/model_comparison.csv")
    print(metrics_df.to_string(index=False))
    if rca_metrics is not None:
        print("Saved RCA metrics to results/rca_metrics.csv")
        print(pd.DataFrame([rca_metrics]).to_string(index=False))


if __name__ == "__main__":
    run_evaluation()
