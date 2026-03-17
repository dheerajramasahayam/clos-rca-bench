from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.chdir(ROOT_DIR)

from anomaly_detection_model.lstm_model import train_model as train_synthetic_lstm
from anomaly_detection_model.train_cisco_real import main as train_cisco_anomaly
from anomaly_detection_model.train_real import train_real_model
from anomaly_detection_model.transformer_model import train_transformer
from dataset.builder import build_dataset
from results.evaluate_topology_benchmark import main as run_topology_benchmark
from root_cause_analysis.rca_model import train_gnn as train_synthetic_rca
from root_cause_analysis.train_cisco_real import main as train_cisco_rca


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    description: str
    dataset_name: str
    runner: Callable[[], None]
    expected_outputs: tuple[Path, ...]


def _train_synthetic_pipeline() -> None:
    train_synthetic_lstm()
    train_transformer()
    train_synthetic_rca()


def _train_gct_pipeline() -> None:
    train_real_model()


def _train_cisco_real_pipeline() -> None:
    train_cisco_anomaly()
    train_cisco_rca()


def _train_topology_pipeline() -> None:
    # The topology benchmark training path is coupled to the benchmark evaluator
    # because it trains the baselines and graph variants before writing leaderboard outputs.
    run_topology_benchmark()


PIPELINE_SPECS: dict[str, PipelineSpec] = {
    "synthetic": PipelineSpec(
        name="synthetic",
        description="Train anomaly detectors and RCA models on the synthetic telemetry benchmark.",
        dataset_name="synthetic",
        runner=_train_synthetic_pipeline,
        expected_outputs=(
            Path("results/lstm_model.pth"),
            Path("results/transformer_model.pth"),
            Path("results/rca_model.pth"),
        ),
    ),
    "gct": PipelineSpec(
        name="gct",
        description="Train the real-telemetry LSTM baseline on the Google Cluster Trace sample.",
        dataset_name="gct",
        runner=_train_gct_pipeline,
        expected_outputs=(Path("results/real/lstm_real_model.pth"),),
    ),
    "cisco-real": PipelineSpec(
        name="cisco-real",
        description="Train anomaly detection and RCA models on the Cisco event-window dataset.",
        dataset_name="cisco-real",
        runner=_train_cisco_real_pipeline,
        expected_outputs=(
            Path("results/cisco_lstm_model.pth"),
            Path("results/cisco_transformer_model.pth"),
            Path("results/cisco_rca_model.pth"),
        ),
    ),
    "topology-benchmark": PipelineSpec(
        name="topology-benchmark",
        description="Run the canonical ClosRCA-Bench training pipeline and write benchmark outputs.",
        dataset_name="topology-benchmark",
        runner=_train_topology_pipeline,
        expected_outputs=(
            Path("results/topology_benchmark_stgnn.pth"),
            Path("results/closrca_bench_leaderboard.csv"),
        ),
    ),
}


def run_pipeline(name: str, *, skip_build: bool = False, skip_download: bool = False) -> None:
    if name not in PIPELINE_SPECS:
        raise KeyError(f"Unknown pipeline '{name}'. Choices: {', '.join(sorted(PIPELINE_SPECS))}")

    spec = PIPELINE_SPECS[name]
    print(f"[train] Running '{spec.name}': {spec.description}")
    if not skip_build:
        build_dataset(spec.dataset_name, skip_download=skip_download)
    spec.runner()
    for path in spec.expected_outputs:
        status = "present" if path.exists() else "missing"
        print(f"[train] {status}: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ClosRCA-Bench pipelines.")
    parser.add_argument(
        "--pipeline",
        default="topology-benchmark",
        choices=[*sorted(PIPELINE_SPECS), "all"],
        help="Training pipeline to run. Use 'all' to execute every pipeline.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Assume datasets are already built and skip dataset generation.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="When building Cisco datasets, reuse local raw inputs instead of downloading them.",
    )
    args = parser.parse_args(argv)

    targets = sorted(PIPELINE_SPECS) if args.pipeline == "all" else [args.pipeline]
    for target in targets:
        run_pipeline(target, skip_build=args.skip_build, skip_download=args.skip_download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
