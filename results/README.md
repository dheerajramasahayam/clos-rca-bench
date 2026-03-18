# Results

This directory contains generated benchmark tables and evaluator implementations.

## Canonical ClosRCA-Bench outputs

These files support the main paper, release snapshot, and public leaderboard:

- `closrca_bench_leaderboard.csv`
- `topology_benchmark_anomaly.csv`
- `topology_benchmark_cause.csv`
- `topology_benchmark_target.csv`
- `topology_benchmark_target_slices.csv`
- `topology_benchmark_temporal_tracking.csv`
- `topology_benchmark_temporal_summary.csv`
- `topology_benchmark_multi_failure.csv`
- `topology_benchmark_positioning.csv`
- `topology_benchmark_case_study.csv`
- `topology_benchmark_propagation_traces.csv`
- `topology_benchmark_remediation.csv`
- `topology_benchmark_digital_twin.csv`

## Supplementary study outputs

These files support the 59-node synthetic Clos stress study used to discuss scale, hidden targets, and simultaneous faults:

- `synthetic_scaleup_summary.csv`
- `why_graph_model.csv`

## Legacy evaluation outputs

These files are retained for the older synthetic and Cisco event-window suites exposed by the public CLI:

- `model_comparison.csv`
- `rca_metrics.csv`
- `cisco_model_comparison.csv`
- `cisco_rca_metrics.csv`

## Implementations

- `evaluate_topology_benchmark.py`: canonical public benchmark evaluator
- `topology_research_extensions.py`: temporal RCA, case-study, propagation, and positioning helpers
- `evaluate_scaleup_synthetic.py`: supplementary 59-node stress evaluator
- `evaluate.py` and `evaluate_cisco.py`: retained legacy evaluators

## Local artifacts

Model checkpoints and encoder/scaler binaries are intentionally ignored by git. Regenerate them locally through the training and evaluation entry points rather than treating them as canonical repository artifacts.
