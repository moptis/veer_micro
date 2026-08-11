"""Execute a notebook in place and report any cells that error.

    python scripts/run_notebook.py notebooks/compare_wrg.ipynb

Runs with allow_errors so one bad cell does not hide the rest, then summarises
what failed. Used to verify a notebook actually works rather than assuming it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    nb = nbformat.read(args.notebook, as_version=4)
    NotebookClient(
        nb, timeout=args.timeout, kernel_name="python3",
        resources={"metadata": {"path": str(args.notebook.parent)}},
        allow_errors=True,
    ).execute()
    nbformat.write(nb, args.notebook)

    bad = 0
    for i, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                bad += 1
                name = out.get("ename", "?")
                val = str(out.get("evalue", ""))[:160]
                print(f"  CELL {i}  {name}: {val}")
    print(f"\ncells with errors: {bad} of {len(nb.cells)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
