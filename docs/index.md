# ClosRCA-Bench

ClosRCA-Bench is an open benchmark for topology-grounded root-cause analysis, hidden-target localization, and counterfactual remediation validation in datacenter networks.

## Links

- Repository: [github.com/dheerajramasahayam/clos-rca-bench](https://github.com/dheerajramasahayam/clos-rca-bench)
- Latest release: [v0.1.0](https://github.com/dheerajramasahayam/clos-rca-bench/releases/tag/v0.1.0)
- Benchmark card: [dataset/cisco_topology_benchmark/BENCHMARK_CARD.md](https://github.com/dheerajramasahayam/clos-rca-bench/blob/main/dataset/cisco_topology_benchmark/BENCHMARK_CARD.md)
- Protocol: [benchmark_protocol/README.md](https://github.com/dheerajramasahayam/clos-rca-bench/blob/main/benchmark_protocol/README.md)
- Paper PDF: [paper.pdf](https://github.com/dheerajramasahayam/clos-rca-bench/blob/main/paper.pdf)

## What the benchmark measures

- Binary anomaly detection over graph windows
- Multiclass cause classification over anomalous windows
- Target-device localization for both observed and hidden targets
- Counterfactual remediation quality in a topology digital twin

## Why hidden-target RCA matters

In operational networks, the failing device is not always directly monitored. ClosRCA-Bench makes that case explicit through hidden-target labels such as `leaf3` and `spine4-3464`, forcing models to infer the fault location from neighboring symptoms and topology structure instead of direct telemetry alone.

## Reproduce the benchmark

```bash
python3 -m pip install -r requirements.txt
python3 scripts/build_dataset.py --dataset topology-benchmark
python3 scripts/train_pipeline.py --pipeline topology-benchmark
python3 scripts/run_evaluation.py --suite topology-benchmark
```

![Topology comparison](https://raw.githubusercontent.com/dheerajramasahayam/clos-rca-bench/main/graphs/topology_model_comparison.png)
