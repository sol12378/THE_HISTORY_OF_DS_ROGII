# ROGII - Wellbore Geology Prediction Analysis

Analysis date: 2026-05-25

## Competition Snapshot

- Competition: ROGII - Wellbore Geology Prediction
- URL: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
- Category: Featured
- Reward: 50,000 USD
- Deadline: 2026-08-05 23:59:00
- Current local data source: `data/raw/`

## Task

Predict `TVT` values for each horizontal well after the Prediction Start point.

The official task deck explains:

- Each horizontal well has TVT values to predict at one-foot steps.
- The metric is RMSE over `manualTVT - predictedTVT`.
- Horizontal wells provide `MD`, `X`, `Y`, `Z`, `GR`, and `TVT_input`.
- `TVT_input` is known until the Prediction Start point and missing after it.
- Each horizontal well has an assigned vertical `typewell` with `TVT` and `GR`.
- The core signal is correlating horizontal-well GR behavior with typewell GR behavior on the TVT axis.

## Local Dataset Structure

Downloaded raw data:

- `data/raw/train/`: 773 wells
- `data/raw/test/`: 3 wells
- `data/raw/sample_submission.csv`: 14,151 prediction rows
- `data/raw/AI_wellbore_geology_prediction_task_en.pptx`: task explanation deck

Each train well has:

- `{well_id}__horizontal_well.csv`
- `{well_id}__typewell.csv`
- `{well_id}.png`

Each test well has:

- `{well_id}__horizontal_well.csv`
- `{well_id}__typewell.csv`

## Horizontal Well Schema

Train horizontal well columns:

- `MD`: measured depth
- `X`, `Y`, `Z`: trajectory coordinates
- `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`: geological formation top/depth references, train only
- `TVT`: target, train only
- `GR`: horizontal gamma ray, can be missing
- `TVT_input`: observed TVT until Prediction Start, missing after Prediction Start

Test horizontal well columns:

- `MD`
- `X`, `Y`, `Z`
- `GR`
- `TVT_input`

Important: train-only columns must not be used directly in a normal train/test feature matrix unless equivalent test-side values are generated or predicted.

## Typewell Schema

Train typewell columns:

- `TVT`
- `GR`
- `Geology`

Test typewell columns:

- `TVT`
- `GR`

`Geology` is useful for analysis and possibly as a training-side auxiliary signal, but it is not present for test typewells.

## Dataset Statistics

Horizontal train wells:

- Wells: 773
- Rows: 5,092,255
- Missing `TVT_input`: 3,783,989
- Missing `GR`: 1,507,972
- Prediction Start mask is a contiguous tail for every train well.

Horizontal test wells:

- Wells: 3
- Rows: 19,221
- Missing `TVT_input`: 14,151
- Missing `GR`: 3,784

Sample submission:

- Rows: 14,151
- Columns: `id`, `tvt`
- Submission rows exactly match the missing `TVT_input` rows in test.

## Critical Leakage Observation

The three test wells are also present in `data/raw/train/`:

- `000d7d20`
- `00bbac68`
- `00e12e8b`

For these wells, test horizontal features are exactly equal to the same columns in the train horizontal files, and the train files contain the target `TVT` for the submission rows.

This means the currently downloadable test set can be solved by direct lookup from train targets. Treat this as a leaderboard/data-leak issue, not as a reliable validation strategy. It may produce an excellent public score but does not measure generalization to unseen wells if Kaggle later changes/evaluates hidden data differently.

## Modeling Interpretation

This is not primarily an image classification problem. It is a structured geoscience sequence-alignment/regression problem:

- The horizontal well has a path through 3D space.
- `TVT_input` anchors the known geology before Prediction Start.
- `GR` after Prediction Start gives a noisy, partially missing signature.
- The typewell gives a vertical reference curve in TVT space.
- The model must infer how TVT evolves along MD after Prediction Start.

The strongest non-leak approach should combine:

- anchor extrapolation from last known `TVT_input`
- trajectory geometry from `MD`, `X`, `Y`, `Z`
- horizontal GR before/after Prediction Start
- typewell GR alignment
- well-level CV that holds out whole wells

## Validation Recommendation

Do not use row-random CV.

Primary CV:

- GroupKFold by `well_id`
- Evaluate only rows where `TVT_input` is missing

Secondary validation:

- hold out wells by spatial clusters
- hold out by typewell similarity groups
- compare wells with high and low GR missingness separately

## Baseline Ladder

1. Direct leakage lookup for downloaded test wells
   - Purpose: verify submission format and leaderboard behavior only.
   - Not a real model.

2. Anchor baseline
   - Predict last known `TVT_input` for all future points.
   - Very fast sanity baseline.

3. Linear/trajectory extrapolation
   - Fit local slope before Prediction Start using `MD`, `Z`, and `TVT_input`.
   - Predict future TVT as smooth continuation.

4. Tree baseline
   - Train LightGBM/XGBoost on missing-tail rows from training wells.
   - Features: normalized MD, delta MD from PS, last known TVT, local slope, X/Y/Z deltas, GR statistics, typewell summary.

5. GR alignment baseline
   - Match horizontal GR windows to typewell GR windows along TVT.
   - Use dynamic time warping/correlation-like features.

6. Blend
   - Blend anchor/trajectory/tree/alignment models using OOF.

## High-Value Features

Anchor features:

- last known `TVT_input`
- last known `MD`, `X`, `Y`, `Z`
- delta MD from Prediction Start
- delta Z from Prediction Start
- local pre-PS TVT slope
- pre-PS TVT curvature

Trajectory features:

- horizontal displacement
- vertical displacement
- inclination proxy
- azimuth proxy
- local dogleg/curvature

GR features:

- raw `GR`
- missing-GR indicator
- forward/backward/interpolated GR
- rolling means/std/slopes
- GR percentile within well
- pre-PS vs post-PS GR distribution shift

Typewell features:

- nearest typewell TVT by last known anchor
- typewell GR at candidate TVT
- local typewell GR rolling stats
- typewell geology segment labels for train-side analysis
- correlation between horizontal GR and typewell GR windows

## Risks

- The public downloadable test leak can distort decision-making.
- `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` are train-only and dangerous to use naively.
- GR missingness is substantial and non-uniform.
- Typewell `Geology` is absent in test.
- A model that memorizes well IDs or exact typewell patterns may not generalize.

## Recommended Next Actions

1. Create a leak-lookup submission only to verify the pipeline and baseline public score.
2. Build `data/processed/train_base.parquet` and `test_base.parquet` from all CSVs with `well_id`, row index, and mask flags.
3. Implement GroupKFold by `well_id` on missing-tail rows.
4. Run `exp001_anchor_baseline`.
5. Run `exp002_lgb_anchor_trajectory`.
6. Add GR/typewell alignment features after CV is trusted.
