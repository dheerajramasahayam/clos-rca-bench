# ClosRCA-Bench v0.1.0

This release turns the repository into a maintained public research artifact rather than a one-time code upload.

## Included in this release

- Public benchmark-oriented repository structure
- Benchmark card and fixed-split evaluation protocol
- Dataset build, training, and evaluation CLI entry points
- Example outputs and a notebook demo
- Leaderboard snapshot and figure assets for citation and review

## Canonical reproduction commands

```bash
python3 -m pip install -r requirements.txt
python3 scripts/build_dataset.py --dataset topology-benchmark
python3 scripts/train_pipeline.py --pipeline topology-benchmark
python3 scripts/run_evaluation.py --suite topology-benchmark
```

## Release assets

- `closrca-bench-v0.1.0-snapshot.zip`: benchmark snapshot bundle for reviewers

## Notes

- Code is released under MIT; source data remains subject to upstream repository terms.
- `.zenodo.json` is included so this release can be archived in Zenodo once repository-to-Zenodo linkage is enabled.
