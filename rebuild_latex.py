"""Rebuild LaTeX documents and clean build artifacts.

Usage:
    uv run python rebuild_latex.py manuscript
"""

import subprocess
import sys
from pathlib import Path

TARGETS = {
    "manuscript": {
        "dir": Path("data/docs/manuscript/document"),
        "tex": "manuscript.tex",
    },
}

ARTIFACTS = ["*.aux", "*.log", "*.out", "*.fls", "*.fdb_latexmk",
             "*.synctex.gz", "*.bbl", "*.blg"]


def build(target_name):
    if target_name not in TARGETS:
        print(f"Unknown target: {target_name}")
        print(f"Available targets: {', '.join(TARGETS)}")
        sys.exit(1)

    target = TARGETS[target_name]
    work_dir = target["dir"]
    tex_file = target["tex"]

    if not (work_dir / tex_file).exists():
        print(f"File not found: {work_dir / tex_file}")
        sys.exit(1)

    # Compile twice for cross-references
    for pass_num in (1, 2):
        print(f"  pdflatex pass {pass_num}...")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_file],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  pdflatex failed on pass {pass_num}")
            print(result.stdout[-2000:])
            sys.exit(1)

    for pattern in ARTIFACTS:
        for f in work_dir.glob(pattern):
            f.unlink()

    pdf_path = work_dir / tex_file.replace(".tex", ".pdf")
    print(f"  {pdf_path}")


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip())
        sys.exit(1)
    target_name = sys.argv[1]
    print(f"Building {target_name}...")
    build(target_name)
    print("Done.")


if __name__ == "__main__":
    main()
