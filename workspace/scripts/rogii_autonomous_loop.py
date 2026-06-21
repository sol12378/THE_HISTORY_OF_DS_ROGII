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


WORKSPACE = Path(__file__).resolve().parents[1]
ACTIONS = WORKSPACE / "rogii_actions.yaml"


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
    results = []
    for step in plan:
        action_id = step["action_id"]
        params = step.get("params") or {}
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
            results.append(result.__dict__)
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
            print(json.dumps(failed, ensure_ascii=False, indent=2), flush=True)
            if not args.continue_on_error:
                raise
    status = "completed" if all(r.get("status") == "completed" for r in results) else "completed_with_failures"
    print(json.dumps({"status": status, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
