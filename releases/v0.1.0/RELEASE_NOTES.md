# ClosRCA-Bench v0.1.0

This release turns the repository into a maintained public research artifact rather than a one-time code upload.

## Included in this release

- Public benchmark-oriented repository structure
- Benchmark card and fixed-split evaluation protocol
- Dataset build, training, and evaluation CLI entry points
- Example outputs and a notebook demo
- Temporal root-cause tracking and detection-delay reporting
- Compound-failure slice reporting and a public case-study trace
- Supplementary 59-node synthetic scale-up and simultaneous-fault stress study
- Deployment-pipeline figure for operational RCA placement
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
- Zenodo DOI for this release: [10.5281/zenodo.19059194](https://doi.org/10.5281/zenodo.19059194).
