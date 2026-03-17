# Leaderboard Schema

Use one row per `(benchmark, task, model)` result.

## Required columns

- `Benchmark`
- `Task`
- `Model`
- `PrimaryMetric`
- `Score`

## Canonical tasks

- `AnomalyDetection`
- `CauseClassification`
- `TargetLocalization`
- `CounterfactualRecovery`

## Canonical benchmark name

- `ClosRCA-Bench`
