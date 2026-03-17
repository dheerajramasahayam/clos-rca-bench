import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_telemetry(num_records=10000, anomaly_rate=0.05):
    """
    Generates synthetic datacenter network telemetry data.
    """
    start_time = datetime.now()
    data = []
    
    # Network entities
    routers = [f'router_{i}' for i in range(10)]
    interfaces = [f'eth{i}' for i in range(4)]
    
    for i in range(num_records):
        timestamp = start_time + timedelta(seconds=i)
        router = random.choice(routers)
        interface = random.choice(interfaces)
        
        # Base metrics (normal state)
        cpu_usage = np.random.normal(30, 5)
        latency = np.random.normal(2, 0.5)
        packet_loss = np.random.exponential(0.01)
        throughput = np.random.normal(800, 50)
        
        is_anomaly = 0
        root_cause = 'none'
        
        # Inject anomalies
        if random.random() < anomaly_rate:
            is_anomaly = 1
            choice = random.choice(['congestion', 'misconfig', 'hardware_failure', 'bgp_instability'])
            root_cause = choice
            
            if choice == 'congestion':
                latency += np.random.uniform(50, 200)
                packet_loss += np.random.uniform(2, 10)
                throughput *= np.random.uniform(0.1, 0.5)
            elif choice == 'misconfig':
                packet_loss += np.random.uniform(5, 20)
                throughput = 0
            elif choice == 'hardware_failure':
                cpu_usage += np.random.uniform(40, 60)
                latency += np.random.uniform(100, 500)
                packet_loss = 100
            elif choice == 'bgp_instability':
                latency += np.random.uniform(10, 50)
                packet_loss += np.random.uniform(0.5, 2)
        
        data.append({
            'timestamp': timestamp,
            'router': router,
            'interface': interface,
            'cpu_usage': max(0, min(100, cpu_usage)),
            'latency': max(0, latency),
            'packet_loss': max(0, min(100, packet_loss)),
            'throughput': max(0, throughput),
            'is_anomaly': is_anomaly,
            'root_cause': root_cause
        })
    
    df = pd.DataFrame(data)
    # Ensure directory exists
    os.makedirs('dataset', exist_ok=True)
    df.to_csv('dataset/network_telemetry.csv', index=False)
    print(f"Generated {num_records} records with {df['is_anomaly'].sum()} anomalies.")
    return df

if __name__ == "__main__":
    generate_telemetry()
