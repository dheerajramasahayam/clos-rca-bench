# Dataset Release

This directory is the public data surface for `clos-rca-bench`.

## Included datasets

| Dataset | Source | Builder | Processed outputs |
| :--- | :--- | :--- | :--- |
| Synthetic telemetry | Generated locally for baseline experiments | `python3 scripts/build_dataset.py --dataset synthetic` | `dataset/X.npy`, `dataset/y_anomaly.npy`, `dataset/y_rca.npy` |
| Google Cluster Trace sample | Local CSV samples under `dataset/real/` | `python3 scripts/build_dataset.py --dataset gct` | `dataset/real_processed/` |
| Cisco real event windows | Public Cisco telemetry traces | `python3 scripts/build_dataset.py --dataset cisco-real` | `dataset/cisco_real_processed/` |
| Cisco Clos topology benchmark | Public Cisco telemetry Clos scenarios | `python3 scripts/build_dataset.py --dataset topology-benchmark` | `dataset/cisco_topology_benchmark/processed/` |

## Public-release intent

- Builders live in [dataset/builder.py](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/dataset/builder.py).
- Downloaders stay separate from processed artifacts so the GitHub repo can remain lightweight.
- The canonical benchmark card is [dataset/cisco_topology_benchmark/BENCHMARK_CARD.md](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/dataset/cisco_topology_benchmark/BENCHMARK_CARD.md).

## Notes for GitHub publication

- Keep builders, cards, and protocol docs under version control.
- Prefer downloading raw Cisco telemetry with the provided scripts instead of committing raw archives.
- Keep processed `.npy` tensors and model checkpoints out of git unless you intentionally publish a frozen release archive.
