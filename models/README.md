# Models

This directory documents the model families exposed by the public benchmark repo.

## Canonical artifacts

- The machine-readable registry is [models/catalog.py](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/models/catalog.py).
- Trained checkpoints are written under `results/`.
- The main benchmark model artifact is `results/topology_benchmark_stgnn.pth`.

## Training entry points

```bash
python3 scripts/train_pipeline.py --pipeline synthetic
python3 scripts/train_pipeline.py --pipeline cisco-real
python3 scripts/train_pipeline.py --pipeline topology-benchmark
```

## Evaluation entry points

```bash
python3 scripts/run_evaluation.py --suite synthetic
python3 scripts/run_evaluation.py --suite cisco-real
python3 scripts/run_evaluation.py --suite topology-benchmark
```
