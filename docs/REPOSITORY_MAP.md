# Repository Map

ClosRCA-Bench is organized around a small canonical public surface and a larger implementation layer underneath it.

## Canonical public surface

- `README.md`: benchmark overview, headline results, and reproduction commands
- `paper.tex` and `paper.pdf`: maintained IEEE-style manuscript and compiled paper
- `dataset/`: dataset builders, download helpers, and benchmark card
- `scripts/`: public entry points for build, training, evaluation, and release asset generation
- `benchmark_protocol/`: fixed-split policy, submission schema, and reporting contract
- `models/`: model registry and public model metadata
- `evaluation/`: evaluation suite router used by the public CLI
- `examples/`: lightweight snapshot artifacts intended for quick inspection
- `docs/`: GitHub Pages content and repository-level documentation
- `submission/`: journal-facing submission notes and local upload packaging instructions

## Benchmark implementation layer

- `telemetry_parser/`: Cisco, GCT, and topology parsing logic
- `anomaly_detection_model/`: retained sequence baselines for older suites
- `root_cause_analysis/`: graph and non-graph RCA model implementations
- `remediation_engine/`: action recommendation, validation, and digital-twin logic
- `results/`: evaluator implementations plus generated CSV outputs
- `graphs/`: generated figures for the manuscript and diagnostics

## Canonical versus supplementary versus legacy

- Canonical: `topology-benchmark` evaluation suite and the outputs documented in `results/README.md`
- Supplementary: `scaleup-synthetic` evaluation suite used to study scale, hidden targets, and simultaneous faults
- Legacy: `synthetic` and `cisco-real` suites retained for reproducibility but not treated as the main benchmark claim

## Release discipline

- Raw Cisco topology archives and processed tensors stay out of git and are regenerated locally.
- Release bundles are created with `python3 scripts/prepare_release_assets.py --version v0.1.0`.
- Journal submission bundles are created with `python3 scripts/prepare_tnsm_submission.py`.
- The Markdown paper export path used earlier in the project has been removed; `paper.tex` is the only maintained manuscript source.
