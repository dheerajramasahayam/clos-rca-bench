# IEEE TNSM Submission Guide

This directory isolates the journal-facing submission path from the benchmark implementation.

## Generate the required files

```bash
python3 scripts/prepare_tnsm_submission.py
```

The script writes these local upload artifacts under `submission/artifacts/`:

- `closrca-bench-tnsm-main-document.pdf`: upload as `Main Document - PDF`
- `closrca-bench-tnsm-main-manuscript.zip`: upload as the LaTeX `Main Manuscript` bundle
- `MANIFEST.txt`: quick check of what was packaged

The LaTeX bundle contains only the canonical manuscript inputs:

- `paper.tex`
- `references.bib`
- `build_paper.sh`
- every figure embedded by `paper.tex`

It intentionally excludes supplementary benchmark artifacts.

## Recommended keywords

Select 3 to 10 items from the journal taxonomy. The strongest fit for this manuscript is:

- `Data Center Networks`
- `Fault Management`
- `Service Assurance`
- `Artificial Intelligence and Machine Learning`
- `Monitoring and Measurements`
- `Autonomic and Cognitive Management`

## Portal answers

These recommendations assume the current manuscript has not been previously submitted, posted as a preprint, or derived from a conference version. Change the answers below if that assumption is false.

| Field | Recommended answer |
| :--- | :--- |
| Previously submitted to this journal? | `No, it wasn't submitted previously` |
| Human subjects? | `No, there were no human subjects` |
| Animal subjects? | `Not applicable` |
| Code associated with manuscript? | `Yes, I have code associated with my manuscript` |
| Data associated with manuscript? | `Yes, I have data associated with my article that I will share` |
| Related to rejected/reviewed/withdrawn manuscript? | `No` |
| Extended version of conference publication? | `No` |
| Related uncited author papers? | `No`, assuming all overlapping author work is already cited |
| Identical preprints posted? | `No` |
| Other posted preprints not prior art? | `No` |
| Conflict of interest file? | `None of the authors have a conflict of interest to disclose`, unless a real COI exists |

## Paste-ready short answers

### Scope

```text
This manuscript is squarely within IEEE TNSM because it addresses fault management and service assurance in datacenter networks through telemetry-driven root cause analysis, topology-aware localization, detection delay, and remediation validation. It combines network management, AI/ML, monitoring, and reproducible benchmarking around an operational network-management problem.
```

### Significance

```text
The paper contributes a public, reproducible benchmark for topology-grounded datacenter RCA, an area where evaluation is usually limited to private traces. It enables direct comparison of anomaly detection, cause localization, hidden-target reasoning, temporal tracking, and remediation quality, making future work in self-healing network management easier to validate and reuse.
```

### Closely related IEEE TNSM papers

```text
CauseFormer: Interpretable Anomaly Detection With Stepwise Attention for Cloud Service (2024); MicroNet: Operation Aware Root Cause Identification of Microservice System Anomalies (2024); Robust Procedural Learning for Anomaly Detection and Observability in 5G RAN (2024); Ensemble Graph Attention Networks for Cellular Network Analytics: From Model Creation to Explainability (2025)
```

Verification links:

- `CauseFormer`: <https://dblp.org/rec/journals/tnsm/ZhongLJC24>
- `MicroNet`: <https://ieeexplore.ieee.org/document/10500843/>
- `Robust Procedural Learning`: <https://dblp.org/rec/journals/tnsm/SundqvistBE24>
- `Ensemble Graph Attention Networks`: <https://dblp.org/rec/journals/tnsm/HajduSzucsVKKSL25>

### Distinctiveness relative to prior work

```text
Unlike prior TNSM works that focus on anomaly detection, service observability, or microservice RCA alone, this paper combines public Clos-topology telemetry, hidden-target localization, temporal delay tracking, propagation traces, compound-failure analysis, and counterfactual remediation validation in one reproducible benchmark and evaluation protocol.
```

## Upload checklist

1. Upload `submission/artifacts/closrca-bench-tnsm-main-document.pdf` as the required PDF.
2. Upload `submission/artifacts/closrca-bench-tnsm-main-manuscript.zip` as the LaTeX main manuscript bundle if the portal requests source files.
3. Keep supplementary benchmark snapshots, notebooks, and release archives out of the main manuscript upload.
4. If the portal asks about code or data, point to the public repository and DOI-backed Zenodo artifact already referenced in the manuscript.
