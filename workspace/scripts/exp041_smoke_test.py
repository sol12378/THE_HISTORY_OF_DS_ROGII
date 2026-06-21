#!/usr/bin/env python3
"""Smoke test for exp041: minimal 5-well run to verify implementation."""
import sys
import os
from pathlib import Path

# Add parent to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Set smoke test flag
os.environ["EXP_MODE"] = "smoke"

# Import and run
from scripts.exp041_pf_residual_gbdt import main

if __name__ == "__main__":
    main(smoke_test=True)
