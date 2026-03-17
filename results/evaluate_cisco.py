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


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from anomaly_detection_model.lstm_model import LSTMAnomalyDetector
from anomaly_detection_model.transformer_model import TransformerAnomalyDetector
from root_cause_analysis.rca_model import TemporalGCNClassifier, build_temporal_adjacency


RESULTS_DIR = Path("results")
GRAPHS_DIR = Path("graphs")


def load_cisco_processed():
    data_dir = Path("dataset/cisco_real_processed")
    X = np.load(data_dir / "X_cisco.npy")
    y_anomaly = np.load(data_dir / "y_cisco_anomaly.npy")
    y_rca = np.load(data_dir / "y_cisco_rca.npy")
    test_idx = np.load(data_dir / "test_idx.npy")
    metadata = pd.read_csv(data_dir / "cisco_window_metadata.csv")
    return X, y_anomaly, y_rca, test_idx, metadata


def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FalsePositiveRate": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def save_roc(y_true, scores, output_name, title):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr, tpr, linewidth=2, color="#b45309", label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / output_name, dpi=200)
    plt.close()


def save_cm(y_true, y_pred, output_name, title, labels):
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(6.2, 5.2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / output_name, dpi=200)
    plt.close()


def save_timeline(metadata, y_true):
    timeline = metadata.iloc[np.load("dataset/cisco_real_processed/test_idx.npy")].copy()
    timeline["window_start_dt"] = pd.to_datetime(timeline["window_start"], unit="s", utc=True)
    timeline["is_anomaly"] = y_true

    plt.figure(figsize=(10, 4.5))
    colors = timeline["scenario"].map(
        {
            "baseline": "#94a3b8",
            "bgp_clear": "#b91c1c",
            "port_flap": "#1d4ed8",
            "transceiver_pull": "#15803d",
        }
    )
    plt.scatter(timeline["window_start_dt"], timeline["is_anomaly"], c=colors, s=28)
    plt.yticks([0, 1], ["Normal", "Anomaly"])
    plt.xlabel("Window Start Time (UTC)")
    plt.ylabel("Label")
    plt.title("Cisco Real Telemetry Test Windows")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "cisco_anomaly_timeline.png", dpi=200)
    plt.close()


def evaluate_anomaly_models():
    X, y_anomaly, _, test_idx, metadata = load_cisco_processed()
    X_test = torch.FloatTensor(X[test_idx])
    y_test = y_anomaly[test_idx]

    models = [
        (
            "LSTM",
            LSTMAnomalyDetector(input_dim=X.shape[2], hidden_dim=64, num_layers=2, output_dim=1),
            RESULTS_DIR / "cisco_lstm_model.pth",
        ),
        (
            "Transformer",
            TransformerAnomalyDetector(input_dim=X.shape[2], d_model=64, nhead=4, num_layers=2),
            RESULTS_DIR / "cisco_transformer_model.pth",
        ),
    ]

    rows = []
    for name, model, checkpoint_path in models:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()

        with torch.no_grad():
            scores = model(X_test).cpu().numpy().flatten()
            predictions = (scores > 0.5).astype(int)

        metrics = binary_metrics(y_test, predictions)
        rows.append({"Dataset": "CiscoRealTelemetry", "Model": name, **metrics})
        save_roc(y_test, scores, f"cisco_roc_{name.lower()}.png", f"Cisco ROC - {name}")
        save_cm(
            y_test,
            predictions,
            f"cisco_cm_{name.lower()}.png",
            f"Cisco Confusion Matrix - {name}",
            ["Normal", "Anomaly"],
        )

    save_timeline(metadata, y_test)
    return rows


def evaluate_rca_model():
    X, _, y_rca, test_idx, _ = load_cisco_processed()
    checkpoint = torch.load(RESULTS_DIR / "cisco_rca_model.pth", map_location="cpu")

    mask = y_rca[test_idx] > 0
    X_test = torch.FloatTensor(X[test_idx][mask])
    y_test = y_rca[test_idx][mask] - 1

    model = TemporalGCNClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        adjacency = build_temporal_adjacency(checkpoint["seq_len"])
        predictions = torch.argmax(model(X_test, adjacency), dim=1).cpu().numpy()

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "PrecisionWeighted": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "RecallWeighted": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1Weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "TestWindows": len(y_test),
    }

    label_names = [checkpoint["label_map"][idx] for idx in sorted(checkpoint["label_map"])]
    save_cm(
        y_test,
        predictions,
        "cisco_rca_cm.png",
        "Cisco RCA Confusion Matrix",
        label_names,
    )

    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "cisco_rca_metrics.csv", index=False)
    return metrics


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    GRAPHS_DIR.mkdir(exist_ok=True)

    anomaly_rows = evaluate_anomaly_models()
    anomaly_df = pd.DataFrame(anomaly_rows)
    anomaly_df.to_csv(RESULTS_DIR / "cisco_model_comparison.csv", index=False)

    rca_metrics = evaluate_rca_model()

    print(anomaly_df.to_string(index=False))
    print(pd.DataFrame([rca_metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
