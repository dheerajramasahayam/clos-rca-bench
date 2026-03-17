# ClosRCA-Bench

ClosRCA-Bench is an open, topology-grounded benchmark for datacenter root-cause analysis, target localization, and remediation validation. The repo is structured as a public benchmark release intended for GitHub publication at [github.com/dheerajramasahayam/clos-rca-bench](https://github.com/dheerajramasahayam/clos-rca-bench).

## Public repo layout

- `dataset/`: dataset builders, source downloaders, benchmark cards, and processed-data conventions
- `scripts/`: public CLI entry points for dataset build, training, and evaluation
- `benchmark_protocol/`: fixed-split evaluation protocol and leaderboard schema
- `models/`: model registry and artifact conventions
- `evaluation/`: canonical benchmark evaluation runner

The rest of the repository keeps the underlying implementation modules used by those public entry points:

- `telemetry_parser/`
- `anomaly_detection_model/`
- `root_cause_analysis/`
- `remediation_engine/`
- `results/`

![Cisco Clos benchmark topology](graphs/cisco_clos_topology.png)

## Benchmark summary

- Source: Cisco public telemetry repository scenarios under `12/`
- Scope: 8 Clos-topology scenarios
- Cause families: `bfd_outage`, `blackhole`, `ecmp_change`, `interface_shutdown`
- Nodes: 11 graph nodes
- Features: 30 per node per time step
- Windows: 311 total
- Label mix: 147 normal, 164 anomalous
- Hidden-target cases: `leaf3` and `spine4-3464`

The canonical benchmark card is [dataset/cisco_topology_benchmark/BENCHMARK_CARD.md](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/dataset/cisco_topology_benchmark/BENCHMARK_CARD.md). The official protocol is [benchmark_protocol/README.md](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/benchmark_protocol/README.md).

## Quickstart

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Build the public benchmark dataset:

```bash
python3 scripts/build_dataset.py --dataset topology-benchmark
```

Run the canonical training pipeline:

```bash
python3 scripts/train_pipeline.py --pipeline topology-benchmark
```

Regenerate benchmark evaluation tables:

```bash
python3 scripts/run_evaluation.py --suite topology-benchmark
```

## Current benchmark results

### Held-out anomaly detection

| Model | Accuracy | Precision | Recall | F1 | False Positive Rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9683 | 1.0000 | 0.9394 | 0.9688 | 0.0000 |
| MLP | 0.8413 | 0.8710 | 0.8182 | 0.8438 | 0.1333 |
| STGNN-Full | 0.8095 | 0.8182 | 0.8182 | 0.8182 | 0.2000 |

### Held-out RCA cause classification

| Model | Accuracy | Weighted F1 | Macro F1 |
| :--- | ---: | ---: | ---: |
| RandomForest | 0.9697 | 0.9707 | 0.9603 |
| MLP | 0.9697 | 0.9707 | 0.9603 |
| STGNN-Full | 0.9394 | 0.9532 | 0.7578 |

### Held-out target-device localization

| Model | Accuracy | Weighted F1 |
| :--- | ---: | ---: |
| RandomForest | 1.0000 | 1.0000 |
| MLP | 0.9697 | 0.9702 |
| STGNN-Full | 0.8485 | 0.8380 |
| STGNN-NoTopology | 0.6364 | 0.6566 |
| STGNN-NoTemporal | 0.7273 | 0.6646 |

### Hidden-target slice

| Model | Hidden-target Accuracy |
| :--- | ---: |
| RandomForest | 1.0000 |
| MLP | 1.0000 |
| STGNN-Full | 1.0000 |
| STGNN-NoTopology | 0.0000 |
| STGNN-NoTemporal | 0.9000 |

### Safety-gated remediation

| Metric | Value |
| :--- | ---: |
| Action Match Rate | 0.8485 |
| Safety Pass Rate | 0.8485 |
| Unsafe Blocked Rate | 1.0000 |
| Recovery Eligible Rate | 0.8485 |

### Counterfactual digital twin recovery

| Metric | Value |
| :--- | ---: |
| Recovery Success Rate | 0.8182 |
| Mean Reachability Gain | 0.0260 |
| Mean Blast Radius Reduction | 0.0260 |
| Mean Overload Reduction | 0.7680 |
| Fault Reachability | 0.9740 |
| Recovered Reachability | 1.0000 |

![Topology benchmark comparison](graphs/topology_model_comparison.png)
![Topology digital twin recovery](graphs/topology_digital_twin_recovery.png)

## Key public entry points

- [scripts/build_dataset.py](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/scripts/build_dataset.py)
- [scripts/train_pipeline.py](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/scripts/train_pipeline.py)
- [scripts/run_evaluation.py](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/scripts/run_evaluation.py)
- [dataset/builder.py](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/dataset/builder.py)
- [models/catalog.py](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/models/catalog.py)
- [benchmark_protocol/submission_template.csv](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/benchmark_protocol/submission_template.csv)
- [results/closrca_bench_leaderboard.csv](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/results/closrca_bench_leaderboard.csv)

## Legacy experiments

The older synthetic and Cisco event-window experiments are still available and now sit behind the same public CLI surface:

- `python3 scripts/build_dataset.py --dataset synthetic`
- `python3 scripts/train_pipeline.py --pipeline cisco-real`
- `python3 scripts/run_evaluation.py --suite synthetic`

## Paper and citation

- Manuscript: [paper.tex](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/paper.tex)
- Citation metadata: [CITATION.cff](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/CITATION.cff)
- Archive metadata: [.zenodo.json](/Users/dheerajramasahayam/Desktop/Projects/clos-rca-bench/.zenodo.json)
