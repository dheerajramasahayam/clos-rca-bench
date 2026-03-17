# ClosRCA-Bench

ClosRCA-Bench is an open topology-grounded benchmark for root cause analysis and remediation validation in datacenter networks.

## Source
- Public source: Cisco telemetry repository, Clos-topology scenarios under `12/`
- License: inherits the source repository license and terms
- Construction code:
  - `dataset/download_cisco_topology_benchmark.py`
  - `telemetry_parser/topology_benchmark.py`

## Scope
- Topology nodes: `11`
- Fault families: `4`
  - `bfd_outage`
  - `blackhole`
  - `ecmp_change`
  - `interface_shutdown`
- Windows: `311`
- Hidden targets: `leaf3`, `spine4-3464`

## Tasks
1. Binary anomaly detection over graph windows.
2. Multiclass cause classification over anomalous windows.
3. Target-device localization over anomalous windows.
4. Counterfactual remediation validation in the topology digital twin.

## Tensor format
- `X_topology.npy`: shape `(num_windows, 6, 11, 30)`
  - `6`: 60-second windows at 10-second cadence
  - `11`: topology nodes
  - `30`: per-node features
- Labels:
  - `y_topology_anomaly.npy`
  - `y_topology_cause.npy`
  - `y_topology_target.npy`
- Splits:
  - `train_idx.npy`
  - `val_idx.npy`
  - `test_idx.npy`

## Feature groups
- Interface data rate
- Interface generic counters
- BFD session state
- BGP process state
- FIB drop counters
- Interface administrative and operational state
- CPU utilization
- Observation mask

## Evaluation files
- `results/topology_benchmark_anomaly.csv`
- `results/topology_benchmark_cause.csv`
- `results/topology_benchmark_target.csv`
- `results/topology_benchmark_target_slices.csv`
- `results/topology_benchmark_remediation.csv`
- `results/topology_benchmark_digital_twin.csv`

## Recommended citation targets
- Benchmark paper: `paper.pdf`
- Benchmark metadata: `CITATION.cff`

## Reproduction
```bash
python3 dataset/download_cisco_topology_benchmark.py
python3 telemetry_parser/topology_benchmark.py
python3 results/evaluate_topology_benchmark.py
```
