# ROGII Autonomous Experiment Plan

作成日: 2026-06-20

対象コンペ: ROGII - Wellbore Geology Prediction

この文書は、ROGII を `autoresearch` 自立研究基盤で実験するための実験計画と、local LLM + scripts のみで完全自律実行を始めるために不足している接続部を整理する。ROGII は NeuroGolf 2026 ではない。well 単位の地質系列回帰問題として扱う。

## 1. 研究方針

ROGII は、horizontal well の Prediction Start 以降の `TVT` を予測する問題である。入力には `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input` と、対応する typewell の `TVT`, `GR` がある。評価対象は `TVT_input` が欠損している tail rows で、指標は RMSE。

最大の注意点は public test/train overlap である。引き継ぎ元の分析では、公開 test 3 wells が train 側にも存在し、train 側には submission 対象行の `TVT` が含まれる。この lookup は提出形式や public LB 挙動の較正には使えるが、generalization model の根拠にはしない。

したがって、実験を次の二つに分離する。

| Lane | 目的 | 判断基準 |
|---|---|---|
| leak-safe lane | unseen wells に転移する実モデルの開発 | GroupKFold by `well_id` の OOF RMSE、per-well error、fold stability |
| public-artifact lane | public LB に強い notebook/fork/artifact の再現・監査 | provenance、leak risk、kernel 再現性、LB 着地 |

local LLM の役割は、任意コードを書き散らすことではなく、履歴・brief・OOF 診断を読んで、許可された script/action space から次の実験を選ぶことである。

overlap wells の扱いを全レーン共通ルールとして先に固定する。`exp102` で検出した public test/train overlap wells は、leak-safe lane の OOF RMSE 計算から常に除外する。除外できない場合でも、必ず独立 slice として分離し、overlap-included と overlap-excluded の両 RMSE を併記する。leak-safe lane の本線 CV 値には overlap-excluded を採用する。

## 2. 基盤上の配置

ROGII は以下に統合済み。

- `competitions/rogii-wellbore-geology-prediction/config.yaml`
- `competitions/rogii-wellbore-geology-prediction/research_brief.yaml`
- `competitions/rogii-wellbore-geology-prediction/workspace/`
- `competitions/rogii-wellbore-geology-prediction/workspace/scripts/`
- `competitions/rogii-wellbore-geology-prediction/workspace/experiments/`
- `competitions/rogii-wellbore-geology-prediction/workspace/ROGII_kaggle_obsidian/`

raw Kaggle data は git 管理外に置く。推奨配置は `{data_root}/rogii-wellbore-geology-prediction/raw/`。引き継ぎ workspace 内の `data/` は historical folds や lightweight artifacts を含むが、raw data の唯一の真実にはしない。

## 3. 実験ログ規約

各実験は `workspace/experiments/<exp_id>/` に保存する。

必須成果物:

- `result.json`
- `notes.md`
- `oof.csv` または `oof.parquet` または `train_oof.npy`
- `per_well.csv`
- `config.yaml` または実行パラメータ snapshot
- leak classification: `low`, `medium`, `high`, `leak_calibration`
- parent experiment と changed-from-parent

可能なら追加する成果物:

- `feature_importance.csv`
- `component_correlation.csv`
- `fold_scores.csv`
- `submission.csv`
- `kernel-metadata.json`
- Kaggle notebook version / submission ref

`result.json` schema は以下に固定する。成功時も失敗時も machine-readable にし、失敗時は `status: "failed"` と `error` を必ず埋める。

```json
{
  "exp_id": "exp120_anchor_repro",
  "status": "ok | failed",
  "parent": "exp110_group_well_cv_v2",
  "changed_from_parent": "anchor baseline を group_well_v002 fold で再実行",
  "lane": "leak_safe | public_artifact | leak_calibration | format_probe",
  "fold_version": "folds_group_well_v002",
  "cv_rmse_overlap_excluded": 15.91,
  "cv_rmse_overlap_included": null,
  "fold_scores": [15.7, 16.1],
  "leak_classification": "low | medium | high | leak_calibration",
  "artifacts": ["oof.csv", "per_well.csv"],
  "runtime_seconds": 540,
  "error": null
}
```

Obsidian 側では以下を更新する。

- `ROGII_kaggle_obsidian/05_Experiments/EXP_Index.md`
- `ROGII_kaggle_obsidian/06_PDCA/Daily_Log/YYYY-MM-DD.md`
- `ROGII_kaggle_obsidian/06_PDCA/Decision_Log.md`
- `ROGII_kaggle_obsidian/09_Submissions/LB_Tracking.md`

## 4. Phase 0: Data Contract と Safety Gate

目的は、raw data から同じ行順・同じ submission id・同じ evaluation mask を再現できる状態にすること。

実験:

| Exp | 内容 | 合格条件 |
|---|---|---|
| `exp100_data_contract` | train/test/sample submission の inventory を作る | well 数、row 数、submission rows が既知値と一致 |
| `exp101_tail_mask_contract` | `TVT_input` 欠損 tail と submission id を対応付ける | test missing tail rows = 14151 |
| `exp102_leak_overlap_audit` | public test wells と train wells の overlap を検査 | overlap wells を明示し、以後 leak flag に使える |
| `exp103_feature_contract` | train-only danger columns を禁止リスト化 | `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`, `Geology` を本線から遮断 |

成果物:

- `data/processed/well_inventory.parquet`
- `data/processed/train_base.parquet`
- `data/processed/test_base.parquet`
- `experiments/exp100_data_contract/result.json`
- `experiments/exp100_data_contract/schema_report.json`

## 5. Phase 1: CV 固定

主 CV は `well_id` 単位の GroupKFold。評価対象は `TVT_input` が欠損している tail rows のみ。row-random CV は禁止。

副 CV:

- typewell similarity holdout
- spatial cluster holdout
- hidden length bucket
- GR missingness bucket
- hard wells only

実験:

| Exp | 内容 | 合格条件 |
|---|---|---|
| `exp110_group_well_cv_v2` | GroupKFold by `well_id` を再生成 | group leakage なし |
| `exp111_cv_health_report` | fold 間の target/tail/GR missingness を監査 | fold imbalance が説明可能 |
| `exp112_oof_slice_harness` | OOF から per-well/slice report を出す | 全実験で再利用可能、overlap well を独立 slice として分離 |

leak-safe lane の本線 CV は overlap-excluded OOF RMSE を採用する(§1 共通ルール)。overlap well が OOF に混入したまま改善を判断してはならない。

この段階で整備する共通 script(原則は既存の棚卸し・adopt。無い機能のみ追加):

- `scripts/build_base_tables.py`(既存)
- `scripts/make_folds.py` / `scripts/make_folds_strat.py`(既存)
- `scripts/analyze_oof.py`(既存。per-well/slice 出力を OOF slice harness として固定)
- submission check 用 script(既存に無ければ追加。§10 gate を実装)

## 6. Phase 2: Leak-Safe Ladder の再現

新規探索の前に、引き継ぎ履歴の強い階段を再現する。

| Exp | 目的 | 期待水準 |
|---|---|---:|
| `exp120_anchor_repro` | last known `TVT_input` baseline | CV 約 15.91 |
| `exp121_lgb_anchor_trajectory_repro` | anchor + trajectory + GR | CV 約 15.05 |
| `exp122_gr_rolling_repro` | rolling GR features | CV 約 13.81 |
| `exp123_geom_extrap_repro` | geometry extrapolation | CV 約 13.53 |
| `exp124_tree_nn_typewell_blend_repro` | tree/NN/typewell-aware blend | CV 約 13.32 |
| `exp125_particle_filter_repro` | PF tracker | CV 約 11.02 |
| `exp126_pf_geom_blend_repro` | PF + geom + smoothing | CV 約 10.16 |
| `exp127_final_blend_repro` | exp026/exp033 相当 blend | CV 10.06 から 9.98 近辺 |

合格条件は、過去 CV と完全一致することではない。データ・fold・依存ライブラリ差を説明でき、component ranking が概ね一致することを重視する。

## 7. Phase 3: PF / Tracker 改善

履歴上、最も大きく効いたのは particle filter 系である。GR/typewell の直接照合は不安定だったが、trajectory smoothness と likelihood の中で弱い観測として使う PF は有効だった。

実験群:

| Exp | 内容 | 狙い |
|---|---|---|
| `exp130_pf_likelihood_v2` | raw GR, normalized GR, GR derivative, missing-aware likelihood | GR の使い方を安定化 |
| `exp131_pf_transition_v2` | monotonicity, slope prior, `TVT + Z` smoothness, curvature-dependent process noise | drift を抑える |
| `exp132_multiscale_pf_full` | multiple window/scale likelihood を full OOF で検証 | subset overfit を避ける |
| `exp133_pf_failure_classifier` | PF が geom/anchor に負ける well を予測 | blend weight に使う |
| `exp134_pf_failure_aware_blend` | PF/geom/tree の soft routing | hard routing の過学習を避ける |

注意:

- subset tuning は full OOF confirmation なしで採用しない。
- per-well hard routing は過去に過学習しているため、soft feature または blend prior として扱う。
- PF confidence signal は、OOF で target を見ずに作れるものだけを採用する。

## 8. Phase 4: Blend / Stack 改善

ROGII では単一モデルより component complementarity が重要である。PF と geom/tree 系は誤差相関が比較的低く、blend 価値がある。

候補 component:

- anchor
- geom extrapolation
- tree blend
- PF original
- PF multiscale
- PF residual GBDT
- typewell-aware NN
- public artifact predictions
- projection postprocess

blend 規約:

- nested-fold OOF で重みを学習する。
- negative weight は原則禁止。
- fold ごとの重み変動を記録する。
- public artifact は provenance が不明なら public-artifact lane に隔離する。
- CV 改善が小さい場合、fold stability と per-well catastrophic degradation を優先する。

実験:

| Exp | 内容 |
|---|---|
| `exp140_component_correlation_audit` | component error correlation と winner map |
| `exp141_nested_nnls_blend_v2` | NNLS blend の再設計 |
| `exp142_stability_regularized_blend` | fold weight stability を正則化 |
| `exp143_projection_postprocess_repro` | exp072 projection の再現 |
| `exp144_submission_candidate_pack` | submit 候補の生成と gate |

## 9. Phase 5: Public Artifact Lane

LB Tracking では `exp072_proj` が LB 8.280、`exp080 sp45-fleongg fork` が LB 7.625 と記録されている。これらは順位面で重要だが、OOF/provenance/leak risk の監査なしに本線へ混ぜない。

実験:

| Exp | 内容 |
|---|---|
| `exp150_public_artifact_inventory` | notebook, dataset, fork, dependency の出所を記録 |
| `exp151_exp080_reproduce_kernel` | exp080 相当 kernel の再現 |
| `exp152_public_artifact_oof_attempt` | OOF 再構築可能性の確認 |
| `exp153_artifact_leakage_audit` | overlap override, train target use, public-specific logic を監査 |
| `exp154_public_artifact_blend_candidate` | 本線とは別に LB-focused candidate を構築 |

分類:

- `trusted_component`: OOF 再構築可能で leak-safe
- `public_artifact`: provenance はあるが OOF 不完全
- `leak_calibration`: public overlap 等に依存
- `reject`: train target / forbidden column / unexplained public-specific logic に依存

## 10. Submission Policy

このコンペは notebook submission 前提であり、ローカル CSV 直接提出は拒否される履歴がある。local script は submission file と notebook artifact を生成し、実提出は Kaggle kernel version を通す。

提出分類:

| 種別 | 用途 | 本線評価に混ぜるか |
|---|---|---|
| `format_probe` | format/kernel 動作確認 | 混ぜない |
| `leak_calibration` | public overlap / lookup の LB 挙動確認 | 混ぜない |
| `leak_safe_model` | CV 改善した本線モデル | 混ぜる |
| `public_artifact_blend` | LB-focused artifact | 別 lane |

提出前 gate:

- row count = 14151
- columns = `id,tvt`
- id 完全一致
- NaN なし
- daily limit 確認
- leak classification あり
- expected CV/LB relation の記述あり
- user 明示許可あり

## 11. 完全自律実行に足りないもの

現状の `autoresearch` 自律ループは、`task_type.load_data(cfg)` が `X, y, groups, X_test` を返し、`models.registry` のホワイトリスト済み sklearn pipeline を回す設計である。これは SIGNATE NIR のような表形式・signal 行列には合うが、ROGII の well 単位 CSV 群、PF、kernel build、public artifact audit にはそのまま合わない。

**アーキ判断(確定)**: ROGII は共通 `AutonomousLoop` / `models.registry` / `load_data` 契約に**無理に合わせない**。実験実行は独立した `RogiiScriptRunner` で完結させ、これが既存 script を action registry 経由で起動する。`geoscience_sequence` task_type は registry 登録のためだけの最小実装とし、config 検証と「この task は RogiiScriptRunner が処理する」という委譲表明にとどめる(`load_data` で sklearn 行列を返さない)。共通 loop へ ROGII を流し込もうとしない。

**既存資産の前提**: workspace には既に 124 本の script があり(`build_base_tables.py`, `make_folds.py`, `analyze_oof.py`, `exp001`〜`exp072` 系など)、過去 CV(15.91→9.98)はこれらが生んだ。よって以下の不足対応は、原則「新規作成」ではなく「既存 script の棚卸し → action registry への登録(必要時のみ薄い wrapper)」とする。新プレフィックスでの作り直しは再現性を壊すため禁止。

不足を優先度順に整理する。

### P0: 実験開始前に必須

| 不足 | 現状 | 必要な対応 |
|---|---|---|
| Python 実行環境 | `.venv` が `C:\Users\doran\AppData\Local\Programs\Python\Python312\python.exe` を参照しており、現シェルで起動不能 | `.venv` 再作成または bundled/runtime Python へ統一。`requirements.txt` と ROGII `workspace/requirements.txt` を導入 |
| ROGII task_type | `config.yaml` は `geoscience_sequence` だが `task_types.registry` に未登録 | `src/autoresearch/task_types/rogii.py` を最小実装(config 検証 + RogiiScriptRunner への委譲表明のみ。sklearn 行列は返さない)し、registry に登録 |
| raw data presence check | ROGII raw data の有無を共通 harness が検査できない | ROGII 専用 inventory script と harness verify を作る |
| submission check | 共通 `SubmissionSpec` は CSV 形式を見られるが、notebook-only submission 運用までは見ない | CSV check に加え、kernel metadata / output path / Kaggle notebook submit 手順を gate 化 |
| action space | 既存 `models.registry` は chemometrics/signal 向け | ROGII 用の script action registry を作る |

### P1: local LLM 自律ループに必須

| 不足 | 現状 | 必要な対応 |
|---|---|---|
| ROGII ExperimentRunner | `AutonomousLoop` は sklearn pipeline factory 前提 | `RogiiScriptRunner` を作り、許可済み script と config を実行する |
| structured result parser | 歴史 scripts は result schema が統一されていない | `result.json` schema を固定し、失敗時も machine-readable にする |
| script whitelist | local LLM が任意 script を走らせると危険 | `rogii_actions.yaml` に許可 action、引数範囲、timeout、resource budget を定義 |
| per-well diagnostics | OOF slice はあるが全実験必須ではない | runner 後段で `per_well.csv` と slice summary を強制生成 |
| experiment ID allocator | 履歴 exp と新規 exp の衝突リスク | `scripts/rogii_next_exp_id.py` または ledger 管理 |

### P2: 長時間自律運用に必要

| 不足 | 現状 | 必要な対応 |
|---|---|---|
| watchdog | PF/GPU/kernel build が長時間化する | timeout, heartbeat, partial artifact capture |
| local LLM health check | config は `http://localhost:8000/v1` を指すが疎通確認なし | startup probe と fallback planner を実運用化 |
| resource arbiter | GPU TCN/PF/Kaggle kernel build が競合しうる | CPU/GPU/memory budget を action に設定 |
| MLflow/VectorStore integration | 共通 loop にはあるが ROGII script runner と未接続 | result.json を MLflow/Chroma に ingest |
| Kaggle automation boundary | submit はユーザー許可が必要 | build までは自律、push/submit は approval-required action として分離 |

## 12. local LLM + scripts 自律化の設計

ROGII では、LLM に Python を自由生成させるより、許可済み action を選ばせる。

推奨 action schema:

各 action は `lane` を必須フィールドとし、runner が実行時点で lane を強制する(分類を §10 submission gate まで遅延させない)。`script` は既存実体を指す。

```yaml
actions:
  - id: data_contract
    script: scripts/build_base_tables.py
    lane: leak_safe
    args_schema: {}
    timeout_minutes: 20
    writes: [data/processed, experiments]
  - id: reproduce_anchor
    script: scripts/exp001_anchor_baseline.py
    lane: leak_safe
    args_schema:
      fold_version: [folds_group_well_v001, folds_group_well_v002]
    timeout_minutes: 20
  - id: reproduce_pf
    script: scripts/exp022_particle_filter.py
    lane: leak_safe
    args_schema:
      n_particles: [64, 128]
      n_seeds: [100, 500]
    timeout_minutes: 90
  - id: nested_blend
    script: scripts/pdca_blend.py
    lane: leak_safe
    args_schema:
      components: list
    timeout_minutes: 20
  - id: oof_slices
    script: scripts/analyze_oof.py
    lane: leak_safe
    args_schema:
      exp_id: string
    timeout_minutes: 10
```

LLM prompt には以下だけを渡す。

- `research_brief.yaml`
- recent `EXP_Index.md`
- last N `result.json`
- current best CV/LB
- failed directions
- available actions
- resource budget

LLM output は JSON のみ。

```json
{
  "action_id": "reproduce_pf",
  "lane": "leak_safe",
  "hypothesis": "PF reproduction is required before tuning likelihood.",
  "params": {"n_particles": 128, "n_seeds": 500},
  "expected_artifacts": ["result.json", "per_well.csv"],
  "leakage_risk": "low",
  "submit_intent": "none"
}
```

runner は JSON schema を検証し、許可外 action、許可外引数、submit action を拒否する。

## 13. 実験開始判定

local LLM と scripts のみで自律実験を開始できる条件:

- `.venv` または実行 Python が復旧している
- ROGII raw data が配置済み
- `geoscience_sequence` task_type が登録済み
- `data_contract` action が通る
- `group_well_cv` action が通る
- `anchor_repro` が OOF と result.json を出す
- `rogii_actions.yaml` に許可 action が定義済み
- LLM endpoint `http://localhost:8000/v1` が疎通する、または `AUTORESEARCH_PLANNER=grid/priority` で fallback できる
- Kaggle submit は自律 loop から直接実行されず、approval-required として止まる

この条件を満たせば、まず `exp100` から `exp127` までの再現実験を自律ループに渡せる。新規改善は、`exp127_final_blend_repro` が通ってから開始する。

### 目標と打ち切り条件(発散防止)

自律ループは無限に回さない。以下の数値 gate を持つ。

- 目標 CV(leak-safe lane, overlap-excluded OOF RMSE): まず `exp127` 水準(約 9.98)再現、その後の新規改善目標を別途設定する。
- phase 打ち切り: 同一 phase 内で連続 5 実験、best OOF RMSE 改善が 0.5% 未満なら phase を打ち切り、次 phase または別レーンへ切替える。
- 総 budget: 1 自律セッションあたりの総実験数または総時間に上限を設け、超過したら停止して人間判断を仰ぐ。
- public-artifact lane への移行は、leak-safe lane が打ち切り条件に達した後に限る。

## 14. 最初の実装 TODO

最短で自律実験開始へ進むための実装順:

0. 既存 124 script と使用済み exp_id(exp001〜exp072 系等)の台帳を作る(衝突防止)。
1. Python environment を復旧する(`.venv` 再作成 + `requirements.txt` ピン留め。手順をコマンドレベルで本 plan に残す)。
2. `src/autoresearch/task_types/rogii.py` を最小実装で追加する(sklearn 行列は返さず RogiiScriptRunner へ委譲)。
3. `src/autoresearch/task_types/registry.py` に `geoscience_sequence` を登録する。
4. 既存 `build_base_tables.py` を data_contract action として棚卸し・登録する。
5. 既存 `make_folds.py` / `make_folds_strat.py` を fold action として棚卸し・登録する。
6. 既存 `analyze_oof.py` を OOF slice harness として固定し、per-well 出力を保証する。
7. `workspace/rogii_actions.yaml` を作る(各 action に `lane` 必須)。
8. `RogiiScriptRunner` を作る(`result.json` schema 強制、失敗時も machine-readable)。
9. local LLM の action JSON を validate して runner に渡す CLI を作る(`lane` / 許可外 action / submit を拒否)。
10. exp_id allocator(`scripts/rogii_next_exp_id.py` か ledger 管理)を作る。
11. `exp100_data_contract` と `exp120_anchor_repro` を smoke 実行する。

この項目が完了すれば、ROGII は local LLM + scripts だけで安全に自律実験を開始できる。

## 15. 実装済み状態 2026-06-20

以下は実装済み。

- `src/autoresearch/task_types/rogii.py`
- `task_types.registry` への `geoscience_sequence` 登録
- `scripts/rogii_build_base_tables.py`
- `scripts/rogii_make_folds.py`
- `scripts/exp120_anchor_repro.py`
- `scripts/rogii_oof_slices.py`
- `rogii_actions.yaml`
- `src/autoresearch/runtime/rogii_script_runner.py`
- `scripts/rogii_run_action.py`
- `scripts/rogii_autonomous_loop.py`

bootstrap 実行コマンド:

```powershell
D:\Data_Science\autoresearch\.venv\Scripts\python.exe `
  competitions\rogii-wellbore-geology-prediction\workspace\scripts\rogii_autonomous_loop.py `
  --mode bootstrap `
  --python D:\Data_Science\autoresearch\.venv\Scripts\python.exe
```

個別 action 実行:

```powershell
D:\Data_Science\autoresearch\.venv\Scripts\python.exe `
  -m autoresearch.runtime.rogii_script_runner `
  --workspace competitions\rogii-wellbore-geology-prediction\workspace `
  --action-id data_contract `
  --python D:\Data_Science\autoresearch\.venv\Scripts\python.exe
```

smoke 結果:

- `exp100_data_contract`: completed
- `exp110_group_well_cv_v2`: completed
- `exp120_anchor_repro`: completed
- `exp120_anchor_repro_oof_slices`: completed
- anchor CV RMSE: `15.909852870734554`
- sample submission rows: `14151`
- submission gate: passed for `experiments/exp120_anchor_repro/submission.csv`

生成済み主要成果物:

- `data/processed/train_base_v001.parquet`
- `data/processed/test_base_v001.parquet`
- `data/processed/typewell_train_base_v001.parquet`
- `data/processed/typewell_test_base_v001.parquet`
- `data/folds/folds_group_well_v002.csv`
- `experiments/exp100_data_contract/result.json`
- `experiments/exp110_group_well_cv_v2/result.json`
- `experiments/exp120_anchor_repro/result.json`
- `experiments/exp120_anchor_repro/oof.csv`
- `experiments/exp120_anchor_repro/per_well.csv`
- `experiments/exp120_anchor_repro/submission.csv`
- `experiments/exp120_anchor_repro/slice_report.json`

注意:

- 現在の `.venv` は通常 sandbox では起動できないことがある。Codex から実行する場合は権限付き実行が必要だった。
- Kaggle submit はまだ自律 action に入れていない。kernel build までは自律化できるが、push/submit は user approval-required とする。
- local LLM には `rogii_actions.yaml` の action id と JSON params のみを返させる。任意 Python 生成は解禁しない。
