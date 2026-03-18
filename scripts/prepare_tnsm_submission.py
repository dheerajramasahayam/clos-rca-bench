#!/usr/bin/env python3
"""Package the canonical manuscript for IEEE TNSM / ScholarOne submission."""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path


FIGURE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
BIB_RE = re.compile(r"\\bibliography\{([^}]+)\}")


def _parse_tex_dependencies(tex_path: Path) -> list[Path]:
    text = tex_path.read_text(encoding="utf-8")
    dependencies: set[Path] = {
        Path("paper.tex"),
        Path("build_paper.sh"),
    }

    for match in FIGURE_RE.finditer(text):
        dependencies.add(Path(match.group(1)))

    for match in BIB_RE.finditer(text):
        for bib_name in match.group(1).split(","):
            bib_path = Path(bib_name.strip())
            if not bib_path.suffix:
                bib_path = bib_path.with_suffix(".bib")
            dependencies.add(bib_path)

    return sorted(dependencies)


def _write_bundle_readme(staging_dir: Path) -> None:
    readme_text = """ClosRCA-Bench IEEE TNSM main manuscript bundle

This directory contains the canonical LaTeX source for the main manuscript only.
Supplementary benchmark artifacts are intentionally excluded.

Build command:
  tectonic paper.tex

Included inputs:
  - paper.tex
  - references.bib
  - figures referenced directly by paper.tex
  - build_paper.sh

Note:
  The manuscript uses the standard IEEEtran template class. If the submission
  portal attempts to rebuild the source, provide the standard IEEE template
  environment or the journal's LaTeX template package as needed.
"""
    (staging_dir / "README.txt").write_text(readme_text, encoding="utf-8")


def _copy_tree(repo_root: Path, staging_dir: Path, dependencies: list[Path]) -> None:
    for relative_path in dependencies:
        source_path = repo_root / relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"Missing manuscript dependency: {relative_path}")
        destination_path = staging_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def _write_manifest(
    out_dir: Path,
    dependencies: list[Path],
    pdf_name: str,
    archive_name: str,
) -> None:
    lines = [
        "ClosRCA-Bench IEEE TNSM submission artifacts",
        "",
        f"Main document PDF: {pdf_name}",
        f"Main manuscript archive: {archive_name}",
        "",
        "LaTeX archive contents:",
    ]
    lines.extend(f"- {path.as_posix()}" for path in dependencies)
    lines.append("- README.txt")
    (out_dir / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_archive(staging_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(staging_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, arcname=path.relative_to(staging_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create upload-ready IEEE TNSM manuscript artifacts."
    )
    parser.add_argument(
        "--outdir",
        default="submission/artifacts",
        help="Directory where the PDF copy and LaTeX archive will be written.",
    )
    parser.add_argument(
        "--pdf-name",
        default="closrca-bench-tnsm-main-document.pdf",
        help="Filename for the upload-ready manuscript PDF.",
    )
    parser.add_argument(
        "--archive-name",
        default="closrca-bench-tnsm-main-manuscript.zip",
        help="Filename for the upload-ready LaTeX source bundle.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    tex_path = repo_root / "paper.tex"
    pdf_path = repo_root / "paper.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError("paper.pdf is missing. Build the manuscript first.")

    out_dir = repo_root / args.outdir
    staging_dir = out_dir / "latex_bundle"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    dependencies = _parse_tex_dependencies(tex_path)
    _copy_tree(repo_root, staging_dir, dependencies)
    _write_bundle_readme(staging_dir)
    _build_archive(staging_dir, out_dir / args.archive_name)
    shutil.copy2(pdf_path, out_dir / args.pdf_name)
    _write_manifest(out_dir, dependencies, args.pdf_name, args.archive_name)

    print(f"Wrote PDF: {(out_dir / args.pdf_name).relative_to(repo_root)}")
    print(f"Wrote archive: {(out_dir / args.archive_name).relative_to(repo_root)}")
    print(f"Wrote manifest: {(out_dir / 'MANIFEST.txt').relative_to(repo_root)}")


if __name__ == "__main__":
    main()
