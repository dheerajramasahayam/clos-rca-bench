# Evaluation

This directory provides the public benchmark evaluation entry point.

## Canonical runner

```bash
python3 scripts/run_evaluation.py --suite topology-benchmark
```

`topology-benchmark` is the only canonical public benchmark suite. `scaleup-synthetic` is supplementary, and `synthetic` / `cisco-real` are retained legacy suites.

## Supported suites

- `synthetic`
- `cisco-real`
- `topology-benchmark`
- `scaleup-synthetic`
- `all`

## Primary outputs

- `results/topology_benchmark_anomaly.csv`
- `results/topology_benchmark_cause.csv`
- `results/topology_benchmark_target.csv`
- `results/topology_benchmark_target_slices.csv`
- `results/topology_benchmark_remediation.csv`
- `results/topology_benchmark_digital_twin.csv`
- `results/closrca_bench_leaderboard.csv`
- `results/synthetic_scaleup_summary.csv`
- `results/why_graph_model.csv`
