import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob

def preprocess_gct_telemetry(data_dir='dataset/real', window_size=10):
    """
    Parses and preprocesses Google Cluster Trace telemetry for ML models.
    """
    csv_files = glob.glob(os.path.join(data_dir, 'gct_sample_*.csv'))
    all_data = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        print(f"File: {file}, Columns: {df.columns.tolist()}")
        # Strip potential BOM or whitespace from column names
        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
        
        # Use first column for timestamp and second for value if exact names fail
        if 'timestamp' not in df.columns:
            df.rename(columns={df.columns[0]: 'timestamp', df.columns[1]: 'mean_CPU_usage_rate'}, inplace=True)
            
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        all_data.append(df)
        
    full_df = pd.concat(all_data).sort_values('timestamp')
    
    # Drop rows with NaNs in CPU usage
    full_df = full_df.dropna(subset=['mean_CPU_usage_rate'])
    
    # Feature scaling
    features = ['mean_CPU_usage_rate']
    scaler = StandardScaler()
    full_df[features] = scaler.fit_transform(full_df[features])
    
    # Save scaler
    os.makedirs('results/real', exist_ok=True)
    joblib.dump(scaler, 'results/real/gct_scaler.joblib')
    
    # Anomaly detection on real data (Labeling based on threshold for demonstration)
    # In a real scenario, we might have labels, here we use a simple heuristic:
    # Any usage > 90th percentile is an "anomaly" for testing the detector
    threshold = full_df['mean_CPU_usage_rate'].quantile(0.95)
    full_df['is_anomaly'] = (full_df['mean_CPU_usage_rate'] > threshold).astype(int)
    
    X = []
    y = []
    
    data_values = full_df[features].values
    anomaly_values = full_df['is_anomaly'].values
    
    for i in range(len(data_values) - window_size):
        X.append(data_values[i:i+window_size])
        y.append(anomaly_values[i+window_size-1])
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"GCT Preprocessed: X={X.shape}, y={y.shape}")
    
    os.makedirs('dataset/real_processed', exist_ok=True)
    np.save('dataset/real_processed/X_gct.npy', X)
    np.save('dataset/real_processed/y_gct.npy', y)
    
    return X, y

if __name__ == "__main__":
    preprocess_gct_telemetry()
