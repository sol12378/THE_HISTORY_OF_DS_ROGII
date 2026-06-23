#!/usr/bin/env python3
"""Minimal autonomous loop for ROGII approved actions.

Mode `bootstrap` runs the mandatory startup sequence without an LLM. This is the
safe first autonomous execution path. Later modes can feed action JSON produced
by a local LLM into the same runner.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.runtime.rogii_script_runner import run_action
from autoresearch.orchestration.resilience import (
    Manifest,
    append_jsonl,
    write_heartbeat,
)


WORKSPACE = Path(__file__).resolve().parents[1]
ACTIONS = WORKSPACE / "rogii_actions.yaml"
# 自律実行の耐障害性アーティファクト（D-2）。
_AUTO_DIR = WORKSPACE / "outputs" / "autonomous"
MANIFEST_PATH = _AUTO_DIR / "manifest.jsonl"      # 完了アクション台帳（resume）
HEARTBEAT_PATH = _AUTO_DIR / "heartbeat.json"     # 外部 watchdog 用心拍
RESULTS_PATH = _AUTO_DIR / "results.jsonl"        # 逐次フラッシュ（途中死で損失局所化）


def _step_key(step: dict) -> str:
    """アクションの安定キー（action_id + exp-id）。resume のスキップ判定に使う。"""
    params = step.get("params") or {}
    return f"{step['action_id']}:{params.get('exp-id', '')}"


BOOTSTRAP_PLAN = [
    {"action_id": "data_contract", "params": {"exp-id": "exp100_data_contract"}},
    {"action_id": "make_folds", "params": {"exp-id": "exp110_group_well_cv_v2", "n-splits": 5}},
    {"action_id": "anchor_repro", "params": {"exp-id": "exp120_anchor_repro"}},
    {"action_id": "oof_slices", "params": {"exp-id": "exp120_anchor_repro"}},
]


def _load_plan(path: str | None) -> list[dict]:
    if not path:
        return BOOTSTRAP_PLAN
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Plan must be a JSON object or list of objects.")
    return data


def _append_action_to_search_store(params: dict) -> None:
    """完了アクションの result.json を memory search store へ追記する。"""
    exp_id = params.get("exp-id") or params.get("exp_id")
    if not exp_id:
        return
    result_path = WORKSPACE / "experiments" / str(exp_id) / "result.json"
    if not result_path.exists():
        return
    try:
        from autoresearch.config import load_paths
        from autoresearch.memory.workspace_record import (
            append_workspace_record,
            load_result_json,
            record_from_result_json,
            should_skip_runtime_append,
        )

        comp_id = "rogii-wellbore-geology-prediction"
        paths = load_paths(ROOT)
        memory_root = Path(paths["memory_root"])
        if not memory_root.is_absolute():
            memory_root = ROOT / memory_root
        data = load_result_json(result_path)
        if should_skip_runtime_append(comp_id, data, str(exp_id)):
            return  # diagnostic / healthcheck / status run -- keep it out of the search store
        record = record_from_result_json(
            comp_id, data, experiment_id=str(exp_id), exp_dir=result_path.parent
        )
        append_workspace_record(memory_root, comp_id, record, direction="minimize")
    except Exception as exc:
        print(f"[warn] search store append failed for {exp_id}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a safe ROGII autonomous action sequence.")
    parser.add_argument("--mode", choices=["bootstrap", "plan"], default="bootstrap")
    parser.add_argument("--plan-json", default=None, help="JSON plan file for mode=plan")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed actions and continue with the remaining plan.",
    )
    args = parser.parse_args()

    plan = BOOTSTRAP_PLAN if args.mode == "bootstrap" else _load_plan(args.plan_json)
    # resume（D-2）: 既に完了したアクションは台帳に基づきスキップする。
    manifest = Manifest(MANIFEST_PATH)
    done = manifest.done()
    results = []
    total = len(plan)
    for idx, step in enumerate(plan):
        action_id = step["action_id"]
        params = step.get("params") or {}
        key = _step_key(step)
        if key in done:
            print(f"[rogii-autonomous] skip (done) action={action_id} key={key}", flush=True)
            continue
        # heartbeat（D-2）: 外部 watchdog が停滞を検知できるよう毎アクション更新。
        write_heartbeat(HEARTBEAT_PATH, progress=idx, total=total,
                        status="running", current=action_id)
        print(f"[rogii-autonomous] action={action_id} params={params}", flush=True)
        started = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
        try:
            result = run_action(
                workspace=WORKSPACE,
                actions_path=ACTIONS,
                action_id=action_id,
                params=params,
                python_exe=args.python,
            )
            rec = result.__dict__
            results.append(rec)
            append_jsonl(RESULTS_PATH, rec)          # 逐次フラッシュ
            if rec.get("status") == "completed":
                manifest.mark(key, action_id=action_id)  # 完了を台帳へ（resume 用）
                _append_action_to_search_store(params)
        except Exception as exc:
            failed = {
                "action_id": action_id,
                "status": "failed",
                "params": params,
                "started_at": started,
                "finished_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
                "error": str(exc),
                "traceback_tail": traceback.format_exc()[-6000:],
            }
            results.append(failed)
            append_jsonl(RESULTS_PATH, failed)       # 失敗も逐次記録
            print(json.dumps(failed, ensure_ascii=False, indent=2), flush=True)
            if not args.continue_on_error:
                raise
    write_heartbeat(HEARTBEAT_PATH, progress=total, total=total, status="done")
    status = "completed" if all(r.get("status") == "completed" for r in results) else "completed_with_failures"
    print(json.dumps({"status": status, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
