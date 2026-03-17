import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import os
import math

class TransformerAnomalyDetector(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerAnomalyDetector, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model)) # Max length 100
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model*2, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :]) # Take last step
        return self.sigmoid(x)

def train_transformer():
    # Load data
    X = np.load('dataset/X.npy')
    y = np.load('dataset/y_anomaly.npy')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)
    
    input_dim = X.shape[2]
    d_model = 32
    nhead = 4
    num_layers = 2
    num_epochs = 20
    batch_size = 32
    
    model = TransformerAnomalyDetector(input_dim, d_model, nhead, num_layers)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0
        for i in range(0, X_train.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(X_train):.4f}')
            
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted == y_test).sum() / y_test.size(0)
        print(f'Test Accuracy: {accuracy.item():.4f}')
        
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/transformer_model.pth')
    print("Model saved to results/transformer_model.pth")

if __name__ == "__main__":
    train_transformer()
