from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DATA_DIR = Path("dataset/cisco_topology_benchmark/processed")


def load_topology_benchmark():
    X = np.load(DATA_DIR / "X_topology.npy")
    y_anomaly = np.load(DATA_DIR / "y_topology_anomaly.npy")
    y_cause = np.load(DATA_DIR / "y_topology_cause.npy")
    y_target = np.load(DATA_DIR / "y_topology_target.npy")
    adjacency = np.load(DATA_DIR / "adjacency.npy")
    train_idx = np.load(DATA_DIR / "train_idx.npy")
    val_idx = np.load(DATA_DIR / "val_idx.npy")
    test_idx = np.load(DATA_DIR / "test_idx.npy")
    return X, y_anomaly, y_cause, y_target, adjacency, train_idx, val_idx, test_idx


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, node_features, adjacency):
        support = self.linear(node_features)
        return torch.einsum("ij,bjf->bif", adjacency, support)


class SpatioTemporalGraphModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        cause_classes,
        target_classes,
        target_node_indices,
        use_topology=True,
        use_temporal=True,
        dropout=0.2,
    ):
        super().__init__()
        self.use_topology = use_topology
        self.use_temporal = use_temporal
        self.target_node_indices = target_node_indices

        if use_temporal:
            self.temporal = nn.GRU(input_dim, hidden_dim, batch_first=True)
            temporal_dim = hidden_dim
        else:
            self.temporal_projection = nn.Linear(input_dim, hidden_dim)
            temporal_dim = hidden_dim

        self.gcn1 = GraphConvolution(temporal_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.anomaly_head = nn.Linear(hidden_dim, 1)
        self.cause_head = nn.Linear(hidden_dim, cause_classes)
        self.target_none_head = nn.Linear(hidden_dim, 1)
        self.target_node_head = nn.Linear(hidden_dim, 1)

    def encode_nodes(self, x):
        batch_size, seq_len, num_nodes, feature_dim = x.shape
        node_sequences = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, feature_dim)

        if self.use_temporal:
            _, hidden = self.temporal(node_sequences)
            node_embeddings = hidden[-1]
        else:
            node_embeddings = self.temporal_projection(node_sequences.mean(dim=1))

        return node_embeddings.view(batch_size, num_nodes, -1)

    def forward(self, x, adjacency):
        node_embeddings = self.encode_nodes(x)
        adjacency = adjacency if self.use_topology else torch.eye(adjacency.shape[0], device=adjacency.device)

        hidden = self.relu(self.gcn1(node_embeddings, adjacency))
        hidden = self.dropout(hidden)
        hidden = self.relu(self.gcn2(hidden, adjacency))

        graph_embedding = hidden.mean(dim=1)
        target_embeddings = hidden[:, self.target_node_indices, :]
        target_logits = self.target_node_head(target_embeddings).squeeze(-1)
        none_logit = self.target_none_head(graph_embedding)
        return {
            "anomaly_logits": self.anomaly_head(graph_embedding).squeeze(-1),
            "cause_logits": self.cause_head(graph_embedding),
            "target_logits": torch.cat([none_logit, target_logits], dim=1),
        }
