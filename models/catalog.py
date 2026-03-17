from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    benchmark: str
    task: str
    implementation: str
    training_entrypoint: str
    evaluation_entrypoint: str
    artifact_path: Path
    notes: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        slug="synthetic-lstm-anomaly",
        benchmark="SyntheticTelemetry",
        task="AnomalyDetection",
        implementation="anomaly_detection_model.lstm_model.LSTMAnomalyDetector",
        training_entrypoint="python3 scripts/train_pipeline.py --pipeline synthetic",
        evaluation_entrypoint="python3 scripts/run_evaluation.py --suite synthetic",
        artifact_path=Path("results/lstm_model.pth"),
        notes="Sequence baseline for the synthetic telemetry benchmark.",
    ),
    ModelSpec(
        slug="synthetic-transformer-anomaly",
        benchmark="SyntheticTelemetry",
        task="AnomalyDetection",
        implementation="anomaly_detection_model.transformer_model.TransformerAnomalyDetector",
        training_entrypoint="python3 scripts/train_pipeline.py --pipeline synthetic",
        evaluation_entrypoint="python3 scripts/run_evaluation.py --suite synthetic",
        artifact_path=Path("results/transformer_model.pth"),
        notes="Transformer anomaly baseline for the synthetic telemetry benchmark.",
    ),
    ModelSpec(
        slug="synthetic-temporal-gcn-rca",
        benchmark="SyntheticTelemetry",
        task="CauseClassification",
        implementation="root_cause_analysis.rca_model.TemporalGCNClassifier",
        training_entrypoint="python3 scripts/train_pipeline.py --pipeline synthetic",
        evaluation_entrypoint="python3 scripts/run_evaluation.py --suite synthetic",
        artifact_path=Path("results/rca_model.pth"),
        notes="Temporal graph RCA model over telemetry windows.",
    ),
    ModelSpec(
        slug="cisco-real-lstm",
        benchmark="CiscoRealTelemetry",
        task="AnomalyDetection",
        implementation="anomaly_detection_model.lstm_model.LSTMAnomalyDetector",
        training_entrypoint="python3 scripts/train_pipeline.py --pipeline cisco-real",
        evaluation_entrypoint="python3 scripts/run_evaluation.py --suite cisco-real",
        artifact_path=Path("results/cisco_lstm_model.pth"),
        notes="Real Cisco anomaly detector trained on public event traces.",
    ),
    ModelSpec(
        slug="cisco-real-transformer",
        benchmark="CiscoRealTelemetry",
        task="AnomalyDetection",
        implementation="anomaly_detection_model.transformer_model.TransformerAnomalyDetector",
        training_entrypoint="python3 scripts/train_pipeline.py --pipeline cisco-real",
        evaluation_entrypoint="python3 scripts/run_evaluation.py --suite cisco-real",
        artifact_path=Path("results/cisco_transformer_model.pth"),
        notes="Transformer anomaly baseline for public Cisco event traces.",
    ),
    ModelSpec(
        slug="cisco-real-temporal-gcn-rca",
        benchmark="CiscoRealTelemetry",
        task="CauseClassification",
        implementation="root_cause_analysis.rca_model.TemporalGCNClassifier",
        training_entrypoint="python3 scripts/train_pipeline.py --pipeline cisco-real",
        evaluation_entrypoint="python3 scripts/run_evaluation.py --suite cisco-real",
        artifact_path=Path("results/cisco_rca_model.pth"),
        notes="Temporal graph RCA model for Cisco public event windows.",
    ),
    ModelSpec(
        slug="closrca-bench-stgnn-full",
        benchmark="ClosRCA-Bench",
        task="JointAnomalyCauseTarget",
        implementation="root_cause_analysis.topology_rca_model.SpatioTemporalGraphModel",
        training_entrypoint="python3 scripts/train_pipeline.py --pipeline topology-benchmark",
        evaluation_entrypoint="python3 scripts/run_evaluation.py --suite topology-benchmark",
        artifact_path=Path("results/topology_benchmark_stgnn.pth"),
        notes="Canonical topology-aware benchmark model used in the public ClosRCA-Bench leaderboard.",
    ),
)
