# Benchmark Protocol

ClosRCA-Bench is evaluated as a fixed-split benchmark, not an ad hoc experiment.

## Tasks

1. Binary anomaly detection on graph windows.
2. Multiclass cause classification on anomalous windows.
3. Target-device localization on anomalous windows.
4. Counterfactual remediation validation in the topology digital twin.

## Canonical dataset

- Benchmark name: `ClosRCA-Bench`
- Source benchmark card: [dataset/cisco_topology_benchmark/BENCHMARK_CARD.md](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/dataset/cisco_topology_benchmark/BENCHMARK_CARD.md)
- Canonical builder: [scripts/build_dataset.py](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/scripts/build_dataset.py) with `--dataset topology-benchmark`
- Canonical evaluator: [scripts/run_evaluation.py](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/scripts/run_evaluation.py) with `--suite topology-benchmark`

## Fixed splits and metadata

- `train_idx.npy`, `val_idx.npy`, `test_idx.npy` are the official split files.
- `window_metadata.csv` is the authoritative per-window manifest.
- Random-state control is pinned in preprocessing and model training code to `42`.

## Primary metrics

| Task | Primary metric | Secondary metrics |
| :--- | :--- | :--- |
| Anomaly detection | `F1` | Accuracy, Precision, Recall, False Positive Rate |
| Cause classification | `F1Weighted` | Accuracy, PrecisionWeighted, RecallWeighted, F1Macro |
| Target localization | `F1Weighted` | Accuracy, PrecisionWeighted, RecallWeighted, F1Macro |
| Counterfactual recovery | `RecoverySuccessRate` | Reachability gain, blast-radius reduction, overload reduction |

## Hidden-target slice

The benchmark contains hidden-target cases where direct telemetry is absent for the true target.

- Hidden targets: `leaf3`, `spine4-3464`
- Required reporting: `ObservedTargets` and `HiddenTargets`
- Output file: `results/topology_benchmark_target_slices.csv`

## Submission outputs

- Compact leaderboard rows should follow [benchmark_protocol/submission_template.csv](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/benchmark_protocol/submission_template.csv).
- The current repo writes `results/closrca_bench_leaderboard.csv` in that format.
- Full benchmark tables remain under `results/topology_benchmark_*.csv`.

## Evaluation policy

- Do not tune on `test_idx.npy`.
- Report results on the official split files only.
- If you change feature extraction, publish the code and regenerate all official CSV outputs.
