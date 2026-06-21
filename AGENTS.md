# ROGII Agent Notes

This competition is ROGII - Wellbore Geology Prediction, not NeuroGolf 2026.

Before proposing or running experiments, read:

- `README.md`
- `research_brief.yaml`
- `workspace/reports/competition_analysis.md`
- `workspace/ROGII_kaggle_obsidian/05_Experiments/EXP_Index.md`
- `workspace/ROGII_kaggle_obsidian/09_Submissions/LB_Tracking.md`

Core discipline:

- Validate by full well holdout, never row-random CV.
- Score only missing-tail rows where `TVT_input` is absent.
- Separate leak-calibration submissions from leak-safe model development.
- Log the hypothesis, data contract, CV result, LB result if any, leakage risk, and next action for every experiment.
- Keep raw data, OOF predictions, model weights, and submissions out of git unless explicitly intended as lightweight historical artifacts.

Primary research lane:

1. Rebuild the inherited leak-safe OOF ladder: anchor, trajectory, GR rolling, geometry, particle filter, blend.
2. Reproduce the inherited best public artifact in a controlled folder and classify every component by provenance.
3. Use per-well errors to choose the next experiment instead of broad random search.
4. Submit only when the submission gate passes and the expected value relative to the current best is explicit.
