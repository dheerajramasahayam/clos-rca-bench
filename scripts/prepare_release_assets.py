from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.chdir(ROOT_DIR)

TOPOLOGY_PROCESSED_DIR = Path("dataset/cisco_topology_benchmark/processed")
RESULTS_DIR = Path("results")
GRAPHS_DIR = Path("graphs")
EXAMPLES_DIR = Path("examples")
RELEASES_DIR = Path("releases")


def ensure_inputs() -> None:
    required_paths = [
        TOPOLOGY_PROCESSED_DIR / "window_metadata.csv",
        TOPOLOGY_PROCESSED_DIR / "X_topology.npy",
        RESULTS_DIR / "closrca_bench_leaderboard.csv",
        GRAPHS_DIR / "topology_target_cm.png",
        GRAPHS_DIR / "topology_model_comparison.png",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Release assets require existing benchmark outputs. Missing: " + ", ".join(missing)
        )


def write_sample_windows(limit: int = 20) -> Path:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(TOPOLOGY_PROCESSED_DIR / "window_metadata.csv")
    sample_path = EXAMPLES_DIR / "closrca_bench_sample_windows.csv"
    metadata.head(limit).to_csv(sample_path, index=False)
    return sample_path


def write_snapshot_json() -> Path:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(TOPOLOGY_PROCESSED_DIR / "window_metadata.csv")
    leaderboard = pd.read_csv(RESULTS_DIR / "closrca_bench_leaderboard.csv")
    temporal_summary = pd.read_csv(RESULTS_DIR / "topology_benchmark_temporal_summary.csv")
    multi_failure = pd.read_csv(RESULTS_DIR / "topology_benchmark_multi_failure.csv")
    case_study = pd.read_csv(RESULTS_DIR / "topology_benchmark_case_study.csv")
    scaleup_summary_path = RESULTS_DIR / "synthetic_scaleup_summary.csv"
    why_graph_path = RESULTS_DIR / "why_graph_model.csv"
    X = np.load(TOPOLOGY_PROCESSED_DIR / "X_topology.npy")
    y_anomaly = np.load(TOPOLOGY_PROCESSED_DIR / "y_topology_anomaly.npy")
    y_cause = np.load(TOPOLOGY_PROCESSED_DIR / "y_topology_cause.npy")
    y_target = np.load(TOPOLOGY_PROCESSED_DIR / "y_topology_target.npy")

    snapshot = {
        "benchmark": "ClosRCA-Bench",
        "version": "v0.1.0",
        "windows": int(len(metadata)),
        "tensor_shape": list(X.shape),
        "scenario_count": int(metadata["scenario"].nunique()),
        "scenarios": sorted(metadata["scenario"].unique().tolist()),
        "anomaly_counts": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(y_anomaly, return_counts=True))
        },
        "cause_counts": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(y_cause, return_counts=True))
        },
        "target_counts": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(y_target, return_counts=True))
        },
        "top_results": leaderboard.to_dict(orient="records"),
        "temporal_tracking": temporal_summary.to_dict(orient="records"),
        "compound_failure_slice": multi_failure.to_dict(orient="records"),
        "case_study": case_study.to_dict(orient="records"),
    }
    if scaleup_summary_path.exists():
        snapshot["synthetic_scaleup"] = pd.read_csv(scaleup_summary_path).to_dict(orient="records")
    if why_graph_path.exists():
        snapshot["why_graph_model"] = pd.read_csv(why_graph_path).to_dict(orient="records")

    snapshot_path = EXAMPLES_DIR / "closrca_bench_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot_path


def write_leaderboard_png() -> Path:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.read_csv(RESULTS_DIR / "closrca_bench_leaderboard.csv").copy()
    leaderboard["Score"] = leaderboard["Score"].map(lambda value: f"{value:.4f}")

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.axis("off")
    table = ax.table(
        cellText=leaderboard.values,
        colLabels=leaderboard.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.38)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#0f766e")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#cbd5e1")

    fig.suptitle("ClosRCA-Bench Leaderboard Snapshot", fontsize=14, weight="bold", y=0.98)
    output_path = EXAMPLES_DIR / "closrca_bench_leaderboard.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_release_zip(version: str) -> Path:
    release_dir = RELEASES_DIR / version
    release_dir.mkdir(parents=True, exist_ok=True)
    zip_path = release_dir / f"closrca-bench-{version}-snapshot.zip"

    include_paths = [
        Path("dataset/cisco_topology_benchmark/BENCHMARK_CARD.md"),
        Path("benchmark_protocol/README.md"),
        Path("results/closrca_bench_leaderboard.csv"),
        Path("results/topology_benchmark_anomaly.csv"),
        Path("results/topology_benchmark_cause.csv"),
        Path("results/topology_benchmark_target.csv"),
        Path("results/topology_benchmark_target_slices.csv"),
        Path("results/topology_benchmark_temporal_summary.csv"),
        Path("results/topology_benchmark_multi_failure.csv"),
        Path("results/topology_benchmark_positioning.csv"),
        Path("results/topology_benchmark_case_study.csv"),
        Path("results/topology_benchmark_propagation_traces.csv"),
        Path("results/topology_benchmark_remediation.csv"),
        Path("results/topology_benchmark_digital_twin.csv"),
        Path("results/synthetic_scaleup_summary.csv"),
        Path("results/why_graph_model.csv"),
        Path("graphs/topology_model_comparison.png"),
        Path("graphs/topology_target_cm.png"),
        Path("graphs/topology_detection_delay.png"),
        Path("graphs/topology_multi_failure.png"),
        Path("graphs/topology_case_study.png"),
        Path("graphs/synthetic_scaleup_performance.png"),
        Path("graphs/datacenter_rca_deployment_pipeline.png"),
        Path("graphs/topology_digital_twin_recovery.png"),
        Path("examples/closrca_bench_sample_windows.csv"),
        Path("examples/closrca_bench_snapshot.json"),
        Path("examples/closrca_bench_leaderboard.png"),
        Path("releases/v0.1.0/RELEASE_NOTES.md"),
    ]

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in include_paths:
            if path.exists():
                archive.write(path, arcname=path.as_posix())

    return zip_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate public example and release assets.")
    parser.add_argument(
        "--version",
        default="v0.1.0",
        help="Release version label used for the snapshot zip path.",
    )
    args = parser.parse_args(argv)

    ensure_inputs()
    sample_path = write_sample_windows()
    snapshot_path = write_snapshot_json()
    leaderboard_png_path = write_leaderboard_png()
    zip_path = build_release_zip(args.version)

    print(f"[release-assets] wrote {sample_path}")
    print(f"[release-assets] wrote {snapshot_path}")
    print(f"[release-assets] wrote {leaderboard_png_path}")
    print(f"[release-assets] wrote {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
