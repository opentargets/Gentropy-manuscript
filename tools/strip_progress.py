"""Remove Spark progress-bar noise from executed notebooks.

    uv run python tools/strip_progress.py chapters/01-data-preparation/*.ipynb

Spark writes a carriage-return progress bar to stderr on every stage tick, which nbconvert
stores as thousands of stream outputs. They carry no information and bloat the repository.
"""

import json
import sys
from pathlib import Path


def clean(path: Path) -> int:
    """Drop progress-bar stream outputs from one notebook. Returns bytes saved."""
    before = path.stat().st_size
    notebook = json.loads(path.read_text())
    for cell in notebook.get("cells", []):
        outputs = cell.get("outputs")
        if not outputs:
            continue
        cell["outputs"] = [
            output
            for output in outputs
            if not (output.get("output_type") == "stream" and "\r" in "".join(output.get("text", [])))
        ]
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    return before - path.stat().st_size


def main():
    """Clean every notebook named on the command line."""
    for name in sys.argv[1:]:
        saved = clean(Path(name))
        print(f"{name}: {saved / 1024:.0f} kB removed")


if __name__ == "__main__":
    main()
