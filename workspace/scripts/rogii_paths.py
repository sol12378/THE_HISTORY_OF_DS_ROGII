"""Shared path helpers for ROGII workspace scripts."""
from __future__ import annotations

import os
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path("D:/Data_Science/data/rogii-wellbore-geology-prediction/raw")


def raw_dir(value: str | None = None) -> Path:
    return Path(value or os.environ.get("ROGII_RAW_DIR") or DEFAULT_RAW).resolve()


def workspace_path(*parts: str) -> Path:
    return WORKSPACE.joinpath(*parts)


def ensure_workspace_dirs() -> None:
    for rel in [
        ("data", "interim"),
        ("data", "processed"),
        ("data", "folds"),
        ("experiments",),
        ("outputs", "logs"),
    ]:
        workspace_path(*rel).mkdir(parents=True, exist_ok=True)
