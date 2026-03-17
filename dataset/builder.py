from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.chdir(ROOT_DIR)

from dataset.download_cisco_real import main as download_cisco_real
from dataset.download_cisco_topology_benchmark import main as download_cisco_topology_benchmark
from dataset.generate_telemetry import generate_telemetry
from telemetry_parser.cisco_parser import preprocess_cisco_telemetry
from telemetry_parser.gct_parser import preprocess_gct_telemetry
from telemetry_parser.parser import preprocess_telemetry
from telemetry_parser.topology_benchmark import preprocess_topology_benchmark


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    raw_paths: tuple[Path, ...]
    processed_paths: tuple[Path, ...]
    builder: Callable[[bool], None]


def _build_synthetic(_: bool) -> None:
    generate_telemetry()
    preprocess_telemetry()


def _build_gct(_: bool) -> None:
    preprocess_gct_telemetry()


def _build_cisco_real(download: bool) -> None:
    if download:
        download_cisco_real()
    preprocess_cisco_telemetry()


def _build_topology_benchmark(download: bool) -> None:
    if download:
        download_cisco_topology_benchmark()
    preprocess_topology_benchmark()


DATASET_SPECS: dict[str, DatasetSpec] = {
    "synthetic": DatasetSpec(
        name="synthetic",
        description="Generate the baseline synthetic telemetry dataset used for anomaly detection and RCA.",
        raw_paths=(Path("dataset/network_telemetry.csv"),),
        processed_paths=(Path("dataset/X.npy"), Path("dataset/y_anomaly.npy"), Path("dataset/y_rca.npy")),
        builder=_build_synthetic,
    ),
    "gct": DatasetSpec(
        name="gct",
        description="Preprocess the local Google Cluster Trace sample files for anomaly experiments.",
        raw_paths=(Path("dataset/real"),),
        processed_paths=(Path("dataset/real_processed/X_gct.npy"), Path("dataset/real_processed/y_gct.npy")),
        builder=_build_gct,
    ),
    "cisco-real": DatasetSpec(
        name="cisco-real",
        description="Download and preprocess the public Cisco event-window telemetry traces.",
        raw_paths=(Path("dataset/cisco_real"),),
        processed_paths=(
            Path("dataset/cisco_real_processed/X_cisco.npy"),
            Path("dataset/cisco_real_processed/y_cisco_anomaly.npy"),
            Path("dataset/cisco_real_processed/y_cisco_rca.npy"),
        ),
        builder=_build_cisco_real,
    ),
    "topology-benchmark": DatasetSpec(
        name="topology-benchmark",
        description="Build ClosRCA-Bench from public Clos-topology scenarios, events, and CDP maps.",
        raw_paths=(Path("dataset/cisco_topology_benchmark/raw"),),
        processed_paths=(
            Path("dataset/cisco_topology_benchmark/processed/X_topology.npy"),
            Path("dataset/cisco_topology_benchmark/processed/y_topology_anomaly.npy"),
            Path("dataset/cisco_topology_benchmark/processed/y_topology_cause.npy"),
            Path("dataset/cisco_topology_benchmark/processed/y_topology_target.npy"),
        ),
        builder=_build_topology_benchmark,
    ),
}


def list_dataset_specs() -> Iterable[DatasetSpec]:
    return DATASET_SPECS.values()


def build_dataset(name: str, *, skip_download: bool = False) -> None:
    if name not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset '{name}'. Choices: {', '.join(sorted(DATASET_SPECS))}")

    spec = DATASET_SPECS[name]
    print(f"[dataset] Building '{spec.name}': {spec.description}")
    spec.builder(not skip_download)
    for path in spec.processed_paths:
        status = "present" if path.exists() else "missing"
        print(f"[dataset] {status}: {path}")


def run_builds(dataset_names: Sequence[str], *, skip_download: bool = False) -> None:
    for name in dataset_names:
        build_dataset(name, skip_download=skip_download)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build public datasets for ClosRCA-Bench.")
    parser.add_argument(
        "--dataset",
        default="topology-benchmark",
        choices=[*sorted(DATASET_SPECS), "all"],
        help="Dataset to build. Use 'all' to build every available dataset.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse local raw inputs instead of downloading Cisco source files again.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available dataset builders and exit.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for spec in list_dataset_specs():
            print(f"{spec.name}: {spec.description}")
        return 0

    targets = sorted(DATASET_SPECS) if args.dataset == "all" else [args.dataset]
    run_builds(targets, skip_download=args.skip_download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
