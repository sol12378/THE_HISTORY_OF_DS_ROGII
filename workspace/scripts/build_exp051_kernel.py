#!/usr/bin/env python3
"""build_exp051_kernel — generate the self-contained blend driver kernel.

Embeds the PROVEN exp026 (PF×geom, LB 8.672) and v11-infer (GBDT meta-stack)
code bodies as base64, runs each as an isolated subprocess, and blends:
    final = 0.438 * artifact(v11) + 0.562 * exp026
(weights from OOF nested-CV: 9.27 vs exp026 10.108, all-fold consistent, leak-free)
"""
from __future__ import annotations
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP026 = ROOT / "kaggle_notebooks/exp026_pf_geom_blend/rogii_exp026_pf_geom_blend.py"
V11 = ROOT / "kaggle_notebooks/exp051_artifact_pf_blend/_v11_infer.py"
OUT = ROOT / "kaggle_notebooks/exp051_artifact_pf_blend/rogii_exp051.py"

exp026_b64 = base64.b64encode(EXP026.read_bytes()).decode()
v11_b64 = base64.b64encode(V11.read_bytes()).decode()

driver = '''"""ROGII exp051 — v11 artifact (GBDT meta-stack) x exp026 (PF x geom) blend.

final_tvt = 0.438 * artifact_v11 + 0.562 * exp026
Weights from OOF nested-fold NNLS: nested CV 9.27 (exp026 alone 10.108, all-fold
consistent, leak-free). Both components run as isolated subprocesses using their
PROVEN code (exp026 = LB 8.672; v11-infer = thbdh5765 public, TabICL-free, CPU).
Self-contained CPU kernel. dataset: thbdh5765/rogii-v11-fresh-artifacts.
"""
import os, sys, base64, subprocess, shutil
from pathlib import Path
import pandas as pd
import numpy as np

W_ART = 0.438
W_EXP026 = 0.562

EXP026_B64 = "%s"
V11_B64 = "%s"

KAGGLE = Path("/kaggle/input").exists()
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
TMP = WORK / "_exp051_components"
TMP.mkdir(parents=True, exist_ok=True)


def find_input_dir() -> Path:
    for root in (Path("/kaggle/input"), Path("data/raw"), Path("data"), Path.cwd()):
        if root.exists():
            hits = list(root.rglob("sample_submission.csv"))
            if hits:
                return hits[0].parent
    raise SystemExit("[exp051] data dir not found")


def find_artifact_dir():
    roots = [Path("/kaggle/input"), Path("/tmp/artifacts")]
    for root in roots:
        if not root.exists():
            continue
        for man in root.rglob("manifest.json"):
            if (man.parent / "models").exists():
                return man.parent
    return None


DATA = find_input_dir()
ART = find_artifact_dir()
print(f"[exp051] KAGGLE={KAGGLE} DATA={DATA} ART={ART} WORK={WORK}", flush=True)

# write proven component code
(TMP / "exp026.py").write_bytes(base64.b64decode(EXP026_B64))
(TMP / "v11.py").write_bytes(base64.b64decode(V11_B64))

# ---- Component B: exp026 (PF x geom) ----
print("[exp051] running exp026 (PF x geom)...", flush=True)
env026 = dict(os.environ)
exp026_cwd = os.getcwd() if not KAGGLE else str(TMP)
subprocess.run([sys.executable, str(TMP / "exp026.py")], check=True, env=env026, cwd=exp026_cwd)
# exp026 writes to /kaggle/working/submission.csv (Kaggle) or ./submission.csv (cwd)
cand = [Path("/kaggle/working/submission.csv"), Path(exp026_cwd) / "submission.csv", WORK / "submission.csv"]
exp026_src = next((c for c in cand if c.exists()), None)
if exp026_src is None:
    raise SystemExit("[exp051] exp026 submission not found")
shutil.move(str(exp026_src), str(TMP / "exp026_sub.csv"))
print(f"[exp051] exp026 done -> {TMP/'exp026_sub.csv'}", flush=True)

# ---- Component A: v11 artifact ----
art_sub = None
if ART is not None:
    print("[exp051] running v11 artifact inference...", flush=True)
    envv = dict(os.environ)
    envv["ROGII_DATA_DIR"] = str(DATA)
    envv["ROGII_ARTIFACT_DIR"] = str(ART)
    envv["ROGII_OUTPUT_DIR"] = str(TMP / "v11out")
    envv["ROGII_INFERENCE_ONLY"] = "1"
    envv["ROGII_SAVE_ARTIFACTS"] = "0"
    envv["ROGII_RUN_TABICL"] = "0"
    (TMP / "v11out").mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([sys.executable, str(TMP / "v11.py")], check=True, env=envv)
        art_sub = TMP / "v11out" / "submission.csv"
        if not art_sub.exists():
            print("[exp051] WARN v11 produced no submission", flush=True)
            art_sub = None
    except subprocess.CalledProcessError as e:
        print(f"[exp051] WARN v11 failed ({e}); falling back to exp026-only", flush=True)
        art_sub = None
else:
    print("[exp051] WARN artifact dir not found; exp026-only", flush=True)

# ---- Blend ----
sample = pd.read_csv(DATA / "sample_submission.csv")[["id"]]
exp026_df = pd.read_csv(TMP / "exp026_sub.csv").rename(columns={"tvt": "t_exp026"})
out = sample.merge(exp026_df, on="id", how="left")
if art_sub is not None:
    art_df = pd.read_csv(art_sub).rename(columns={"tvt": "t_art"})
    out = out.merge(art_df, on="id", how="left")
    out["t_art"] = out["t_art"].fillna(out["t_exp026"])
    out["tvt"] = W_ART * out["t_art"] + W_EXP026 * out["t_exp026"]
    print(f"[exp051] BLEND {W_ART}*artifact + {W_EXP026}*exp026", flush=True)
else:
    out["tvt"] = out["t_exp026"]
    print("[exp051] exp026-only (artifact unavailable)", flush=True)

assert not out["tvt"].isna().any(), "NaN in final submission"
out[["id", "tvt"]].to_csv(WORK / "submission.csv", index=False)
print(f"[exp051] wrote {WORK/'submission.csv'} rows={len(out)} "
      f"tvt[{out.tvt.min():.1f},{out.tvt.max():.1f}]", flush=True)
''' % (exp026_b64, v11_b64)

OUT.write_text(driver)
print(f"wrote {OUT} ({len(driver)} chars, exp026 {len(exp026_b64)}b64, v11 {len(v11_b64)}b64)")
