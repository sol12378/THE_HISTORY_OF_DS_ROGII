#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


WORKSPACE = Path(__file__).resolve().parents[1]
KERNEL_DIR = WORKSPACE / "kaggle_notebooks"
EXP_ROOT = WORKSPACE / "experiments"

NOTEBOOKS = [
    ("exp090_lightningv08_lb7168", 7.168),
    ("exp091_baidalinadilzhan_lb7201", 7.201),
    ("exp092_kokinnwakashuu_dual_pipeline_v16", None),
    ("exp093_curvecowboy_lb7295_public_rebuild", 7.295),
    ("exp094_omprakashpy_public_gold_fallback", None),
]

MARKERS = [
    "ROGII_GOLD_PROFILE",
    "selector_well_code",
    "apply_selector_variant",
    "run_particle_filter",
    "run_pf_lik_ensemble_scales",
    "run_beam_ensemble",
    "sp45_fleongg",
    "guarded_contact_override",
    "_gold_calibrate_well",
    "_gold_candidate_pool",
    "_gold_reapply_guarded_contact_override",
    "submission_gold_prefix",
    "version2_submission_audit",
]


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _script_path(kernel_path: Path, meta: dict) -> Path:
    code_file = meta.get("code_file") or ""
    if code_file.endswith(".ipynb"):
        return kernel_path / code_file.replace(".ipynb", ".py")
    return kernel_path / code_file


def _extract_profiles(text: str) -> list[str]:
    vals = set(re.findall(r"ROGII_GOLD_PROFILE[\"']?\]?\s*=\s*[\"']([a-zA-Z_]+)[\"']", text))
    vals.update(re.findall(r"_GOLD_PROFILE\s*=\s*[^\\n]*get\([^\\n]*,\s*[\"']([a-zA-Z_]+)[\"']", text))
    return sorted(vals)


def _summarize_notebook(name: str, reported_lb: float | None) -> dict:
    kernel_path = KERNEL_DIR / name
    meta_path = kernel_path / "kernel-metadata.json"
    response_path = kernel_path / "kernel_pull_response.json"
    meta = _load_json(meta_path)
    script = _script_path(kernel_path, meta)
    text = script.read_text(encoding="utf-8")
    lines = text.splitlines()
    marker_counts = {m: text.count(m) for m in MARKERS}
    to_csv = sorted(set(re.findall(r"to_csv\((?:_WORK / |CFG.OUT/|[^)]*)?[\"']([^\"']+\\.csv)[\"']", text)))
    imports = sorted(set(re.findall(r"^(?:from|import)\s+([A-Za-z_][A-Za-z0-9_\\.]+)", text, re.M)))
    return {
        "exp_id": name,
        "reported_lb": reported_lb,
        "kernel_id": meta.get("id"),
        "title": meta.get("title"),
        "last_run_time": meta.get("last_run_time"),
        "enable_gpu": meta.get("enable_gpu"),
        "enable_internet": meta.get("enable_internet"),
        "dataset_sources": meta.get("dataset_sources") or [],
        "competition_sources": meta.get("competition_sources") or [],
        "script_file": str(script.relative_to(WORKSPACE)),
        "response_bytes": response_path.stat().st_size if response_path.exists() else None,
        "script_bytes": script.stat().st_size,
        "script_lines": len(lines),
        "script_sha256": _sha256(script),
        "profiles": _extract_profiles(text),
        "marker_counts": marker_counts,
        "csv_outputs": to_csv,
        "top_imports": imports[:80],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", default="exp095_public_lb71_intake")
    args = parser.parse_args()

    exp_dir = EXP_ROOT / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    notebooks = [_summarize_notebook(name, lb) for name, lb in NOTEBOOKS]
    by_hash: dict[str, list[str]] = defaultdict(list)
    for nb in notebooks:
        by_hash[nb["script_sha256"]].append(nb["exp_id"])

    required_datasets = sorted({ds for nb in notebooks for ds in nb["dataset_sources"]})
    local_external = WORKSPACE / "data" / "external"
    local_external_status = {
        ds: (local_external / ds.split("/")[-1]).exists()
        for ds in required_datasets
    }
    raw_dir = Path("D:/Data_Science/data/rogii-wellbore-geology-prediction/raw")
    raw_status = {
        "raw_dir": str(raw_dir),
        "exists": raw_dir.exists(),
        "sample_submission": (raw_dir / "sample_submission.csv").exists(),
        "train_dir": (raw_dir / "train").exists(),
        "test_dir": (raw_dir / "test").exists(),
    }

    result = {
        "exp_id": args.exp_id,
        "created_at": _now(),
        "status": "completed",
        "task": "public_lb71_notebook_intake",
        "submission_floor_lb": 7.1,
        "best_reference_lb": 7.168,
        "notebooks": notebooks,
        "equivalence_groups": [
            {"script_sha256": h, "members": members}
            for h, members in sorted(by_hash.items(), key=lambda kv: kv[1][0])
        ],
        "required_datasets": required_datasets,
        "local_external_status": local_external_status,
        "raw_status": raw_status,
        "next_actions": [
            "Mirror or mount the three Kaggle dataset sources locally.",
            "Run group A/B/C notebooks in a Kaggle-compatible local wrapper or Kaggle notebook.",
            "Collect submission.csv and audit files per group.",
            "Compare group A/B/C submissions per test well and row segment.",
        ],
    }

    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for nb in notebooks:
        rows.append(
            "| {exp_id} | {reported_lb} | {title} | {script_lines} | {hash8} | {profiles} |".format(
                exp_id=nb["exp_id"],
                reported_lb="" if nb["reported_lb"] is None else nb["reported_lb"],
                title=(nb["title"] or "").replace("|", "/"),
                script_lines=nb["script_lines"],
                hash8=nb["script_sha256"][:8],
                profiles=", ".join(nb["profiles"]),
            )
        )
    notes = "\n".join(
        [
            f"# {args.exp_id}",
            "",
            "## Summary",
            "",
            f"- Best public reference LB: 7.168",
            f"- Submission floor: 7.1",
            f"- Source families: {len(by_hash)}",
            f"- Required datasets: {', '.join(required_datasets)}",
            "",
            "## Notebook inventory",
            "",
            "| Exp | LB | Title | Lines | Hash | Profiles |",
            "|---|---:|---|---:|---|---|",
            *rows,
            "",
            "## Next",
            "",
            "- Mirror external Kaggle datasets.",
            "- Execute one representative per source family: A=exp090, B=exp091, C=exp092.",
            "- Build per-well prediction-difference ledger.",
        ]
    )
    (exp_dir / "notes.md").write_text(notes + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
