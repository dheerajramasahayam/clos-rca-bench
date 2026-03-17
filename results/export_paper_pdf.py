import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
PAPER_PATH = ROOT / "research_paper.md"
OUTPUT_PATH = ROOT / "research_paper.pdf"


def markdown_to_lines(markdown_text, width=92):
    lines = []
    for raw_line in markdown_text.splitlines():
        stripped = raw_line.rstrip()
        if not stripped:
            lines.append("")
            continue

        if stripped.startswith("#"):
            lines.append(stripped.replace("#", "").strip().upper())
            lines.append("")
            continue

        if stripped.startswith("|"):
            lines.append(stripped)
            continue

        if stripped.startswith("- ") or stripped[:3].isdigit():
            wrapped = textwrap.wrap(
                stripped,
                width=width,
                subsequent_indent="  ",
                replace_whitespace=False,
            )
        else:
            wrapped = textwrap.wrap(
                stripped,
                width=width,
                replace_whitespace=False,
            )

        lines.extend(wrapped or [""])
    return lines


def add_text_pages(pdf, lines, lines_per_page=42):
    for start in range(0, len(lines), lines_per_page):
        chunk = lines[start : start + lines_per_page]
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(
            0.06,
            0.96,
            "\n".join(chunk),
            ha="left",
            va="top",
            family="DejaVu Sans",
            fontsize=10.5,
            linespacing=1.35,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def add_figure_page(pdf, image_path, title):
    if not image_path.exists():
        return

    image = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    ax.imshow(image)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    markdown_text = PAPER_PATH.read_text()
    lines = markdown_to_lines(markdown_text)

    figure_pages = [
        (ROOT / "graphs" / "datacenter_telemetry_pipeline.png", "Datacenter Telemetry Pipeline"),
        (ROOT / "graphs" / "ai_model_architecture.png", "AI Model Architecture"),
        (ROOT / "graphs" / "anomaly_detection_timeline.png", "Anomaly Detection Timeline"),
        (ROOT / "graphs" / "roc_lstm.png", "ROC Curve"),
        (ROOT / "graphs" / "cm_lstm.png", "Confusion Matrix"),
    ]

    with PdfPages(OUTPUT_PATH) as pdf:
        add_text_pages(pdf, lines)
        for image_path, title in figure_pages:
            add_figure_page(pdf, image_path, title)

    print(f"Saved PDF to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
