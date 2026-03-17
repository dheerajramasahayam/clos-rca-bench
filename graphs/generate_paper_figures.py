import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
GRAPHS_DIR = ROOT / "graphs"


def add_box(ax, x, y, w, h, title, subtitle, facecolor):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.5,
        edgecolor="#0f172a",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.63,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        x + w / 2,
        y + h * 0.34,
        subtitle,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#1f2937",
    )


def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color="#0f172a",
        )
    )


def build_pipeline_figure():
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, "Datacenter Telemetry", "SNMP, NetFlow,\nBGP logs, counters", "#dbeafe"),
        (0.20, "Aggregation Layer", "Parser, windowing,\nfeature scaling", "#dcfce7"),
        (0.37, "AI Anomaly Detection", "LSTM and Transformer\nsequence models", "#fef3c7"),
        (0.54, "Graph RCA Engine", "Temporal graph reasoning\nfor failure class", "#fde68a"),
        (0.71, "Remediation Engine", "Rollback, reroute,\nquarantine actions", "#fecaca"),
        (0.88, "Automated Recovery", "Operator guidance or\nclosed-loop healing", "#e9d5ff"),
    ]

    width = 0.12
    height = 0.38
    y = 0.31
    for x, title, subtitle, color in boxes:
        add_box(ax, x, y, width, height, title, subtitle, color)

    for left, right in zip(boxes, boxes[1:]):
        add_arrow(ax, (left[0] + width, y + height / 2), (right[0], y + height / 2))

    ax.text(
        0.02,
        0.90,
        "Self-Healing Datacenter Network Pipeline",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.02,
        0.84,
        "From raw telemetry to AI-assisted diagnosis and remediation.",
        fontsize=10.5,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "datacenter_telemetry_pipeline.png", dpi=220)
    fig.savefig(ROOT / "architecture.png", dpi=220)
    plt.close(fig)


def build_model_figure():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        0.08,
        0.62,
        0.22,
        0.18,
        "Telemetry Window",
        "10-step sequence of\nlatency, loss, CPU,\nthroughput, device IDs",
        "#dbeafe",
    )
    add_box(
        ax,
        0.40,
        0.73,
        0.2,
        0.14,
        "LSTM Encoder",
        "Temporal state tracking",
        "#dcfce7",
    )
    add_box(
        ax,
        0.40,
        0.47,
        0.2,
        0.14,
        "Transformer Encoder",
        "Attention over long-range\ntelemetry correlations",
        "#fef3c7",
    )
    add_box(
        ax,
        0.72,
        0.60,
        0.2,
        0.18,
        "Temporal GCN RCA",
        "Graph message passing across\nthe anomalous observation window",
        "#fde68a",
    )
    add_box(
        ax,
        0.72,
        0.30,
        0.2,
        0.14,
        "Policy / Recovery",
        "Remediation recommendation\nor automated response",
        "#fecaca",
    )

    add_arrow(ax, (0.30, 0.71), (0.40, 0.80))
    add_arrow(ax, (0.30, 0.71), (0.40, 0.54))
    add_arrow(ax, (0.60, 0.80), (0.72, 0.69))
    add_arrow(ax, (0.60, 0.54), (0.72, 0.69))
    add_arrow(ax, (0.82, 0.60), (0.82, 0.44))

    ax.text(
        0.08,
        0.92,
        "AI Model Stack",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.08,
        0.87,
        "Sequence encoders surface anomalies; the graph layer classifies likely failure causes.",
        fontsize=10.5,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "ai_model_architecture.png", dpi=220)
    plt.close(fig)


def main():
    GRAPHS_DIR.mkdir(exist_ok=True)
    build_pipeline_figure()
    build_model_figure()
    print("Saved conceptual figures to graphs/ and architecture.png")


if __name__ == "__main__":
    main()
