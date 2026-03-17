import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def preprocess_telemetry(file_path='dataset/network_telemetry.csv', window_size=10):
    """
    Parses and preprocesses telemetry data for ML models.
    """
    df = pd.read_csv(file_path)
    
    # Sort by timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Encode categorical features
    le_router = LabelEncoder()
    df['router_enc'] = le_router.fit_transform(df['router'])
    
    le_interface = LabelEncoder()
    df['interface_enc'] = le_interface.fit_transform(df['interface'])
    
    # Features to scale
    features = ['cpu_usage', 'latency', 'packet_loss', 'throughput', 'router_enc', 'interface_enc']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Save scalers and encoders
    os.makedirs('results', exist_ok=True)
    joblib.dump(scaler, 'results/scaler.joblib')
    joblib.dump(le_router, 'results/le_router.joblib')
    joblib.dump(le_interface, 'results/le_interface.joblib')
    
    # Create sequences for LSTM/Transformer
    X = []
    y_anomaly = []
    y_rca = [] # Root Cause Analysis target
    
    # Map root causes to integers
    rca_map = {'none': 0, 'congestion': 1, 'misconfig': 2, 'hardware_failure': 3, 'bgp_instability': 4}
    df['rca_enc'] = df['root_cause'].map(rca_map)
    
    data_values = df[features].values
    anomaly_values = df['is_anomaly'].values
    rca_values = df['rca_enc'].values
    
    for i in range(len(data_values) - window_size):
        X.append(data_values[i:i+window_size])
        # Target is the anomaly status of the LAST element in the window
        y_anomaly.append(anomaly_values[i+window_size-1])
        y_rca.append(rca_values[i+window_size-1])
        
    X = np.array(X)
    y_anomaly = np.array(y_anomaly)
    y_rca = np.array(y_rca)
    
    print(f"Preprocessed data shape: X={X.shape}, y_anomaly={y_anomaly.shape}, y_rca={y_rca.shape}")
    
    # Save processed data
    np.save('dataset/X.npy', X)
    np.save('dataset/y_anomaly.npy', y_anomaly)
    np.save('dataset/y_rca.npy', y_rca)
    
    return X, y_anomaly, y_rca

if __name__ == "__main__":
    preprocess_telemetry()
