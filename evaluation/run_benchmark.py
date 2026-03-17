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

from results.evaluate import run_evaluation as evaluate_synthetic
from results.evaluate_cisco import main as evaluate_cisco_real
from results.evaluate_topology_benchmark import main as evaluate_topology_benchmark


@dataclass(frozen=True)
class EvaluationSpec:
    name: str
    description: str
    runner: Callable[[], None]
    key_outputs: tuple[Path, ...]


EVALUATION_SPECS: dict[str, EvaluationSpec] = {
    "synthetic": EvaluationSpec(
        name="synthetic",
        description="Evaluate the synthetic anomaly and RCA baselines.",
        runner=evaluate_synthetic,
        key_outputs=(Path("results/rca_metrics.csv"),),
    ),
    "cisco-real": EvaluationSpec(
        name="cisco-real",
        description="Evaluate the Cisco real-telemetry anomaly and RCA models.",
        runner=evaluate_cisco_real,
        key_outputs=(Path("results/cisco_model_comparison.csv"), Path("results/cisco_rca_metrics.csv")),
    ),
    "topology-benchmark": EvaluationSpec(
        name="topology-benchmark",
        description="Run the canonical ClosRCA-Bench evaluator and regenerate leaderboard outputs.",
        runner=evaluate_topology_benchmark,
        key_outputs=(
            Path("results/topology_benchmark_anomaly.csv"),
            Path("results/topology_benchmark_cause.csv"),
            Path("results/topology_benchmark_target.csv"),
            Path("results/closrca_bench_leaderboard.csv"),
        ),
    ),
}


def run_suite(name: str) -> None:
    if name not in EVALUATION_SPECS:
        raise KeyError(f"Unknown evaluation suite '{name}'. Choices: {', '.join(sorted(EVALUATION_SPECS))}")

    spec = EVALUATION_SPECS[name]
    print(f"[eval] Running '{spec.name}': {spec.description}")
    spec.runner()
    for path in spec.key_outputs:
        status = "present" if path.exists() else "missing"
        print(f"[eval] {status}: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ClosRCA-Bench evaluation suites.")
    parser.add_argument(
        "--suite",
        default="topology-benchmark",
        choices=[*sorted(EVALUATION_SPECS), "all"],
        help="Evaluation suite to run. Use 'all' to regenerate every benchmark table.",
    )
    args = parser.parse_args(argv)

    targets = sorted(EVALUATION_SPECS) if args.suite == "all" else [args.suite]
    for target in targets:
        run_suite(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
