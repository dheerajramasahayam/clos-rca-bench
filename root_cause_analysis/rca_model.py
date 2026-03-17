import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_temporal_adjacency(seq_len):
    """Create a normalized chain graph over the telemetry window."""
    adjacency = torch.eye(seq_len)
    for idx in range(seq_len - 1):
        adjacency[idx, idx + 1] = 1.0
        adjacency[idx + 1, idx] = 1.0

    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
    return degree_inv_sqrt @ adjacency @ degree_inv_sqrt


def split_rca_dataset(test_size=0.2, random_state=42):
    X = np.load("dataset/X.npy")
    y_rca = np.load("dataset/y_rca.npy")

    anomaly_mask = y_rca > 0
    X_rca = X[anomaly_mask]
    y_labels = y_rca[anomaly_mask] - 1

    if len(X_rca) == 0:
        raise ValueError("No anomalous windows are available for RCA training.")

    return train_test_split(
        X_rca,
        y_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=y_labels,
    )


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features, adjacency):
        support = self.linear(node_features)
        return torch.einsum("ij,bjf->bif", adjacency, support)


class TemporalGCNClassifier(nn.Module):
    """
    A lightweight graph model over the telemetry window.

    Each time step is treated as a node and message passing captures how the
    anomalous pattern evolves across the observation window before classifying
    the likely root cause.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.conv1 = GraphConvolution(input_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adjacency):
        x = self.relu(self.conv1(x, adjacency))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, adjacency))
        x = x.mean(dim=1)
        return self.classifier(x)


def train_gnn(num_epochs=80, learning_rate=0.001, hidden_dim=64):
    set_seed()

    X_train, X_test, y_train, y_test = split_rca_dataset()

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    seq_len = X_train.shape[1]
    input_dim = X_train.shape[2]
    output_dim = len(torch.unique(y_train))

    adjacency = build_temporal_adjacency(seq_len)
    model = TemporalGCNClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training temporal GCN RCA model on {len(X_train)} anomaly windows...")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train, adjacency)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test, adjacency)
        predictions = torch.argmax(test_logits, dim=1).cpu().numpy()
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

    print(
        "RCA Metrics | "
        f"Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision_weighted']:.4f}, "
        f"Recall: {metrics['recall_weighted']:.4f}, "
        f"F1: {metrics['f1_weighted']:.4f}"
    )

    os.makedirs("results", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "seq_len": seq_len,
            "metrics": metrics,
            "label_map": {
                0: "congestion",
                1: "misconfig",
                2: "hardware_failure",
                3: "bgp_instability",
            },
        },
        "results/rca_model.pth",
    )
    print("RCA checkpoint saved to results/rca_model.pth")


if __name__ == "__main__":
    train_gnn()
