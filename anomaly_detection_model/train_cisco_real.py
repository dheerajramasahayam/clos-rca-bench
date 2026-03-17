from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from anomaly_detection_model.lstm_model import LSTMAnomalyDetector
from anomaly_detection_model.transformer_model import TransformerAnomalyDetector


def load_cisco_split():
    data_dir = Path("dataset/cisco_real_processed")
    X = np.load(data_dir / "X_cisco.npy")
    y = np.load(data_dir / "y_cisco_anomaly.npy")
    train_idx = np.load(data_dir / "train_idx.npy")
    test_idx = np.load(data_dir / "test_idx.npy")
    return X, y, train_idx, test_idx


def train_model(model_name, model, epochs=30, batch_size=32, learning_rate=0.001):
    X, y, train_idx, test_idx = load_cisco_split()

    X_train = torch.FloatTensor(X[train_idx])
    y_train = torch.FloatTensor(y[train_idx]).view(-1, 1)
    X_test = torch.FloatTensor(X[test_idx])
    y_test = torch.FloatTensor(y[test_idx]).view(-1, 1)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))

        for batch_start in range(0, X_train.size(0), batch_size):
            indices = permutation[batch_start : batch_start + batch_size]
            batch_x = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"{model_name} epoch {epoch + 1}/{epochs} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predictions = (model(X_test) > 0.5).float()
        accuracy = (predictions == y_test).sum().item() / y_test.size(0)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / f"cisco_{model_name.lower()}_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"{model_name} Cisco accuracy={accuracy:.4f}")
    print(f"Saved {model_name} checkpoint to {checkpoint_path}")


def main():
    X, _, _, _ = load_cisco_split()
    input_dim = X.shape[2]

    train_model(
        "LSTM",
        LSTMAnomalyDetector(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=1),
        epochs=30,
    )
    train_model(
        "Transformer",
        TransformerAnomalyDetector(input_dim=input_dim, d_model=64, nhead=4, num_layers=2),
        epochs=30,
    )


if __name__ == "__main__":
    main()
