#!/usr/bin/env python3
"""
exp049: v11 artifact ローカル実行スクリプト
環境変数を設定してnotebookコードを実行。
"""

import os
import sys
from pathlib import Path

# ===== 環境設定 =====
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ARTIFACT_DIR = Path("/tmp/artifacts/thbdh5765_rogii-v11-fresh-artifacts")
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp049_v11_local"

# ディレクトリ作成
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 環境変数を先に設定
os.environ["ROGII_DATA_DIR"] = str(DATA_DIR)
os.environ["ROGII_ARTIFACT_DIR"] = str(ARTIFACT_DIR)
os.environ["ROGII_OUTPUT_DIR"] = str(OUTPUT_DIR)
os.environ["ROGII_INFERENCE_ONLY"] = "1"
os.environ["ROGII_SAVE_ARTIFACTS"] = "0"
os.environ["ROGII_LGB_DEVICE"] = "cpu"  # CPU-only
os.environ["ROGII_CB_TASK_TYPE"] = "CPU"  # CPU-only

print(f"[exp049] CONFIG")
print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
print(f"  DATA_DIR: {DATA_DIR}")
print(f"  ARTIFACT_DIR: {ARTIFACT_DIR}")
print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  ROGII_INFERENCE_ONLY: {os.environ['ROGII_INFERENCE_ONLY']}")
print(f"  ROGII_LGB_DEVICE: {os.environ['ROGII_LGB_DEVICE']}")
print()

# ===== notebook コード実行 =====
notebook_code_path = Path("/tmp/notebook_code.py")
if not notebook_code_path.exists():
    print("[ERROR] Notebook code not found. Extract first.")
    sys.exit(1)

print(f"[exp049] Executing notebook code ({notebook_code_path.stat().st_size // 1024}KB)...")
with open(notebook_code_path) as f:
    code = f.read()

# Prepare a namespace for execution with all needed globals
namespace = {
    "__name__": "__main__",
    "__file__": str(notebook_code_path),
}

# Run in separate namespace to isolate
exec(compile(code, str(notebook_code_path), 'exec'), namespace)

print(f"\n[exp049] COMPLETE")
print(f"  Check: {OUTPUT_DIR}/submission.csv")

