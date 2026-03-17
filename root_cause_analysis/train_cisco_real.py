from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from root_cause_analysis.rca_model import TemporalGCNClassifier, build_temporal_adjacency


def load_cisco_rca_split():
    data_dir = Path("dataset/cisco_real_processed")
    X = np.load(data_dir / "X_cisco.npy")
    y_rca = np.load(data_dir / "y_cisco_rca.npy")
    train_idx = np.load(data_dir / "train_idx.npy")
    test_idx = np.load(data_dir / "test_idx.npy")

    X_train = X[train_idx]
    y_train = y_rca[train_idx]
    X_test = X[test_idx]
    y_test = y_rca[test_idx]

    train_mask = y_train > 0
    test_mask = y_test > 0

    return (
        X_train[train_mask],
        y_train[train_mask] - 1,
        X_test[test_mask],
        y_test[test_mask] - 1,
    )


def main():
    X_train, y_train, X_test, y_test = load_cisco_rca_split()

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    seq_len = X_train.shape[1]
    input_dim = X_train.shape[2]
    output_dim = len(torch.unique(y_train))

    adjacency = build_temporal_adjacency(seq_len)
    model = TemporalGCNClassifier(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(80):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train, adjacency)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Cisco RCA epoch {epoch + 1}/80 loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(X_test, adjacency), dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision_weighted": precision_score(
            y_true, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, predictions, average="weighted", zero_division=0
        ),
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": 64,
        "output_dim": output_dim,
        "seq_len": seq_len,
        "metrics": metrics,
        "label_map": {
            0: "bgp_clear",
            1: "port_flap",
            2: "transceiver_pull",
        },
    }

    output_path = Path("results/cisco_rca_model.pth")
    torch.save(checkpoint, output_path)
    print(metrics)
    print(f"Saved Cisco RCA checkpoint to {output_path}")


if __name__ == "__main__":
    main()
