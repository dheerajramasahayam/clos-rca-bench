# Notebooks

This folder is reserved for exploratory and presentation-friendly notebooks that complement the scripted pipeline.

Suggested notebook split:
- `01_eda.ipynb`: telemetry profiling, class balance, and fault signatures
- `02_anomaly_comparison.ipynb`: LSTM vs Transformer training curves and threshold analysis
- `03_rca_error_analysis.ipynb`: RCA confusion patterns and remediation review

The repository is currently reproducible through Python scripts in `dataset/`, `telemetry_parser/`, `anomaly_detection_model/`, `root_cause_analysis/`, `graphs/`, and `results/`.
