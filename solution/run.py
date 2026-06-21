"""Thin launcher for inherited ROGII experiments.

The historical project under ../workspace owns the domain-specific scripts. This
entrypoint keeps the autonomous research base oriented around the right
competition and points operators to the shared submission gate.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COMP_DIR = ROOT / "competitions" / "rogii-wellbore-geology-prediction"
WORKSPACE = COMP_DIR / "workspace"
CONFIG = COMP_DIR / "config.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ROGII workspace commands.")
    parser.add_argument(
        "--script",
        default="scripts/run_exp.py",
        help="Workspace-relative script to run, e.g. scripts/run_exp.py",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the script")
    ns = parser.parse_args()

    script = (WORKSPACE / ns.script).resolve()
    if not script.exists():
        print(f"Script not found: {script}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(script), *ns.args]
    print(f"Competition config: {CONFIG}")
    print(f"Workspace: {WORKSPACE}")
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=WORKSPACE)


if __name__ == "__main__":
    raise SystemExit(main())
