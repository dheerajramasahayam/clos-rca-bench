# Scripts

These are the public entry points for the GitHub repository.

## Common commands

```bash
python3 scripts/build_dataset.py --dataset topology-benchmark
python3 scripts/train_pipeline.py --pipeline cisco-real
python3 scripts/run_evaluation.py --suite topology-benchmark
```

## Intent

- `build_dataset.py`: one CLI for dataset creation and public-data regeneration.
- `train_pipeline.py`: one CLI for model training pipelines by benchmark or dataset family.
- `run_evaluation.py`: one CLI for benchmark evaluation and leaderboard regeneration.
