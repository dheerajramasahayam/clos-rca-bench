import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import os
from anomaly_detection_model.lstm_model import LSTMAnomalyDetector

def train_real_model():
    # Load GCT data
    X = np.load('dataset/real_processed/X_gct.npy')
    y = np.load('dataset/real_processed/y_gct.npy')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)
    
    input_dim = X.shape[2]
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    num_epochs = 30 # Slightly more for real noise
    batch_size = 64
    learning_rate = 0.001
    
    model = LSTMAnomalyDetector(input_dim, hidden_dim, num_layers, output_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training LSTM on real GCT dataset...")
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
        print(f'GCT Test Accuracy: {accuracy.item():.4f}')
        
    os.makedirs('results/real', exist_ok=True)
    torch.save(model.state_dict(), 'results/real/lstm_real_model.pth')
    print("Real Model saved to results/real/lstm_real_model.pth")

if __name__ == "__main__":
    train_real_model()
