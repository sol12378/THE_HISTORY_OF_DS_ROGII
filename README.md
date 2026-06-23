---
type: competition
comp_id: rogii-wellbore-geology-prediction
title: ROGII - Wellbore Geology Prediction
platform: kaggle
task_type: geoscience_sequence
status: active
deadline: 2026-08-05
metric: rmse
best_cv: 8.79
best_lb: 7.168
rank:
tags: [kaggle, geoscience_sequence, regression, wellbore, inherited-workspace]
---

# ROGII - Wellbore Geology Prediction

Kaggle workspace integrated into the autonomous research base.

This is not NeuroGolf 2026. Treat it as a geoscience sequence-regression competition: predict missing tail `TVT` values for horizontal wells using anchors, trajectory, gamma ray (`GR`), and assigned typewell curves.

## Current State

- Inherited repository: `workspace/`
- Competition notes: `workspace/reports/competition_analysis.md`
- Experiment index: `workspace/ROGII_kaggle_obsidian/05_Experiments/EXP_Index.md`
- Submission/LB tracking: `workspace/ROGII_kaggle_obsidian/09_Submissions/LB_Tracking.md`
- Best public LB target now tracked: `7.168` from `lightningv08/rogii-lb-7-168`
- Submission floor: only promote candidates that can plausibly compete with the LB `7.1` public-notebook family.

## Operating Rules

- Use `config.yaml` for shared data/submission/CV metadata.
- Keep raw Kaggle data outside git at `{data_root}/rogii-wellbore-geology-prediction/raw/`.
- Primary validation is GroupKFold by `well_id`, evaluated only on rows where `TVT_input` is missing.
- Public test/train overlap is a known leakage issue. Lookup submissions are allowed only for pipeline verification and must be labelled as leakage-calibration, not as a general solution.
- Every experiment should write `result.json`, `notes.md`, and any OOF/per-well diagnostics under `workspace/experiments/<exp_id>/`.

## Fast Start

From `D:/Data_Science/autoresearch`:

```powershell
scripts\runtime_python.cmd -m autoresearch.cli --config competitions\rogii-wellbore-geology-prediction\config.yaml limit-status
scripts\runtime_python.cmd -m autoresearch.cli --config competitions\rogii-wellbore-geology-prediction\config.yaml check-submission <submission.csv>
scripts\runtime_python.cmd competitions\rogii-wellbore-geology-prediction\workspace\scripts\rogii_autonomous_loop.py --mode bootstrap --python C:\Users\doran\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

Historical scripts remain available under `workspace/scripts/`. Prefer reproducing a small, leak-safe OOF experiment before submitting anything new.
