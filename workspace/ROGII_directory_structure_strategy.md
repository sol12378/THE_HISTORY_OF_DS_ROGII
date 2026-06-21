# ROGII - Wellbore Geology Prediction 実験ディレクトリ設計書

作成日: 2026-05-23  
対象: Kaggle「ROGII - Wellbore Geology Prediction」  
想定運用: MacBook Pro / RTX 2080 Super / Google Colab / Claude Code / Codex  
目的: Kaggleコンペでの実験速度・再現性・CV信頼性・OOF管理・最終アンサンブルを最大化する

---

## 0. 設計思想

このディレクトリ構成は、単なる整理整頓ではなく、**Kaggleで勝つための実験管理基盤**として設計する。

先人のKaggle write-upや実務的なデータサイエンスプロジェクト構成から得られる教訓は、主に以下である。

1. **最初に信頼できるCVを作る**
2. **OOFを必ず保存する**
3. **1実験1目的にする**
4. **実験のconfig・結果・submission・notesを必ず残す**
5. **生データは絶対に変更しない**
6. **Notebookは探索用、勝負コードはscript/package化する**
7. **public LBだけで判断せず、CV-LBの乖離を追跡する**
8. **最後は単一モデルではなく、OOFベースでblend/stackする**
9. **Claude Code / Codexが迷わないようにルールと出力形式を固定する**

ROGIIは画像分類コンペというより、**井戸・深度・座標・地質ログを扱う表形式 + 1D系列 + 空間データ予測コンペ**として扱う。  
そのため、画像学習のような重いGPU勝負ではなく、まずは **XGBoost / LightGBM / CatBoost / CV設計 / 特徴量設計 / OOF分析 / Blend** を中心に戦う。

---

## 1. 推奨ディレクトリ構成

```text
rogii-wellbore/
  README.md
  CLAUDE.md
  AGENTS.md
  EXP_SUMMARY.md
  TODO.md
  CHANGELOG.md
  requirements.txt
  pyproject.toml
  .gitignore

  data/
    raw/
    interim/
    processed/
    folds/
    external/
    submissions/
    cache/

  notebooks/
    00_eda_overview.ipynb
    01_reproduce_xgb_starter.ipynb
    02_cv_validation_check.ipynb
    03_feature_ablation.ipynb
    04_oof_error_analysis.ipynb
    05_blend_exploration.ipynb

  src/
    rogii/
      __init__.py

      config/
        default.yaml
        exp001_xgb_baseline.yaml
        exp002_lgb_anchor_delta.yaml
        exp003_cat_typewell.yaml
        exp004_xgb_spatial_rolling.yaml
        exp005_tree_blend.yaml
        exp006_tcn_sequence.yaml

      data/
        load_data.py
        make_dataset.py
        make_folds.py
        validate_schema.py
        split_strategy.py

      features/
        __init__.py
        basic.py
        anchor.py
        depth_series.py
        spatial.py
        typewell.py
        rolling.py
        target.py
        leakage_checks.py

      models/
        __init__.py
        xgb_model.py
        lgb_model.py
        cat_model.py
        tcn_model.py
        baseline_rules.py
        blend.py
        stack.py

      training/
        train_tree.py
        train_nn.py
        train_cv.py
        predict.py
        infer_submission.py

      evaluation/
        metrics.py
        cv_report.py
        oof_analysis.py
        lb_tracking.py
        feature_importance.py
        error_slicing.py

      visualization/
        plot_well.py
        plot_trajectory.py
        plot_typewell.py
        plot_oof_error.py
        plot_feature_importance.py

      utils/
        logger.py
        seed.py
        io.py
        timer.py
        paths.py
        memory.py

  experiments/
    exp001_xgb_baseline/
      config.yaml
      result.json
      cv.csv
      oof.csv
      feature_importance.csv
      train.log
      submission.csv
      notes.md

    exp002_lgb_anchor_delta/
      config.yaml
      result.json
      cv.csv
      oof.csv
      feature_importance.csv
      train.log
      submission.csv
      notes.md

  scripts/
    download_data.sh
    make_folds.py
    run_exp.py
    run_exp_colab.py
    make_submission.py
    blend_submissions.py
    analyze_oof.py
    check_typewell_duplicates.py
    check_trajectory_duplicates.py
    export_kaggle_dataset.py

  reports/
    data_understanding.md
    cv_strategy.md
    feature_ablation.md
    oof_error_analysis.md
    leakage_risk.md
    final_solution.md
    postmortem.md

  outputs/
    figures/
    logs/
    models/
    oof/
    predictions/
    blends/
```

---

## 2. 各ディレクトリの役割

### 2.1 `data/`

```text
data/
  raw/
  interim/
  processed/
  folds/
  external/
  submissions/
  cache/
```

### `data/raw/`

Kaggleからダウンロードした生データをそのまま置く。  
**絶対に編集しない。**

ルール:

- 手作業で中身を変更しない
- 前処理結果を置かない
- Git管理しない
- 破損確認だけ行う

`.gitignore` では以下のようにする。

```gitignore
data/raw/*
data/interim/*
data/processed/*
data/cache/*
outputs/*
experiments/*/oof.csv
experiments/*/submission.csv
experiments/*/*.pkl
experiments/*/*.model
```

### `data/interim/`

前処理途中の一時ファイルを置く。

例:

- parquet化した中間データ
- 井戸単位に整形したデータ
- Typewellを展開したデータ
- 欠損処理前後の比較用データ

### `data/processed/`

学習に使える形まで整えたデータを置く。

例:

```text
train_base.parquet
test_base.parquet
train_anchor_features.parquet
train_typewell_features.parquet
```

### `data/folds/`

CV分割を保存する。

これは非常に重要。  
Kaggleでは、実験ごとにfoldが変わると比較が壊れる。

例:

```text
folds_group_well_v001.csv
folds_group_typewell_v001.csv
folds_spatial_holdout_v001.csv
```

各foldファイルは以下の列を持つ。

```text
row_id, well_id, typewell_id, fold
```

### `data/submissions/`

Kaggleに提出したCSVの保管場所。

例:

```text
sub_2026-05-23_exp001_xgb_baseline.csv
sub_2026-05-24_blend_v003.csv
```

Kaggle提出後は、LBスコアを `EXP_SUMMARY.md` と `reports/lb_tracking.md` に記録する。

---

## 3. `notebooks/` の役割

Notebookは探索・確認・可視化専用にする。  
本番の学習コードをNotebookに閉じ込めない。

```text
notebooks/
  00_eda_overview.ipynb
  01_reproduce_xgb_starter.ipynb
  02_cv_validation_check.ipynb
  03_feature_ablation.ipynb
  04_oof_error_analysis.ipynb
  05_blend_exploration.ipynb
```

### Notebook運用ルール

1. Notebookで良いアイデアを見つける
2. その処理を `src/rogii/features/` または `src/rogii/models/` に移す
3. `scripts/run_exp.py` から再現できるようにする
4. `experiments/expXXX/` に結果を保存する

Notebookに残してよいもの:

- EDA
- plot
- 仮説確認
- OOF誤差の可視化
- blend重みの探索

Notebookに残してはいけないもの:

- 最終学習パイプライン
- 重要な特徴量生成の唯一の実装
- 提出用CSV生成の唯一の実装

---

## 4. `src/rogii/` の設計

### 4.1 `config/`

すべての実験はconfigで管理する。

```yaml
exp_id: exp002_lgb_anchor_delta
seed: 42

data:
  train_path: data/processed/train_base.parquet
  test_path: data/processed/test_base.parquet
  fold_path: data/folds/folds_group_well_v001.csv

features:
  sets:
    - basic
    - anchor
    - depth_series

target:
  mode: delta_from_anchor
  column: TVT

model:
  name: lightgbm
  params:
    objective: regression
    learning_rate: 0.03
    num_leaves: 64
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 1
    n_estimators: 3000

training:
  n_folds: 5
  early_stopping_rounds: 100
  max_train_minutes: 20

output:
  save_oof: true
  save_submission: true
  save_feature_importance: true
```

config化する理由:

- Claude/Codexが差分実験を作りやすい
- 再現性が上がる
- 実験比較が容易になる
- 「何を変えた実験か」が明確になる

---

## 5. 特徴量ディレクトリ設計

```text
features/
  basic.py
  anchor.py
  depth_series.py
  spatial.py
  typewell.py
  rolling.py
  target.py
  leakage_checks.py
```

### 5.1 `basic.py`

基本特徴量。

例:

- MD
- X
- Y
- Z
- well_id encoding
- typewell_id encoding
- row order
- numeric columnsそのまま

### 5.2 `anchor.py`

ROGIIで最重要候補。

考え方:

```text
予測値 = last_known_value + predicted_delta
```

特徴量例:

- last_known_TVT
- last_known_GR
- delta_MD_from_anchor
- delta_Z_from_anchor
- slope_from_anchor
- distance_from_last_known
- relative_position_after_anchor

### 5.3 `depth_series.py`

深度方向の系列特徴量。

例:

- normalized_MD
- MD_rank_within_well
- MD_percentile
- local_depth_index
- segment_position
- depth_gap

### 5.4 `spatial.py`

X/Y/Z座標から作る空間特徴量。

例:

- horizontal_displacement
- vertical_displacement
- azimuth
- inclination
- curvature
- dogleg-like feature
- distance_to_typewell
- distance_to_nearest_known_point

### 5.5 `typewell.py`

Typewell関連の特徴量。

例:

- typewell_hash
- duplicate_typewell_group
- typewell_log_mean
- typewell_log_std
- typewell_trend
- nearest_typewell_value
- typewell_depth_alignment_feature

DiscussionでTypewellの重複・類似に関する指摘があるため、ここは優先度が高い。

### 5.6 `rolling.py`

深度方向のlag/rolling特徴量。

例:

- lag_1
- lag_3
- diff_1
- rolling_mean_5
- rolling_std_5
- rolling_slope_5
- expanding_mean
- local_gradient

### 5.7 `target.py`

target変換を管理する。

例:

- direct target
- delta from anchor
- residual from simple baseline
- local slope target
- smoothed target

target変換は実験結果に強く影響するため、必ずconfigと紐付ける。

---

## 6. CV設計

ROGIIではCVが最重要。  
row random splitは原則禁止。

### 6.1 CV候補

```text
1. GroupKFold by well_id
2. GroupKFold by typewell_id
3. Spatial holdout
4. Time/depth-like holdout
5. Public LB correlation check
```

### 6.2 保存すべきCV情報

`cv.csv` には以下を保存する。

```text
fold, score, n_train, n_valid, train_time_sec, best_iteration
```

`result.json` には平均と標準偏差を保存する。

```json
{
  "cv_mean": 12.345,
  "cv_std": 0.456,
  "fold_scores": [12.1, 12.7, 12.2, 12.6, 12.1],
  "cv_strategy": "GroupKFold_by_well",
  "lb_score": null
}
```

### 6.3 CV-LB Tracking

`reports/lb_tracking.md` に以下を記録する。

```markdown
| Date | Exp | CV | LB | Gap | Notes |
|---|---:|---:|---:|---:|---|
| 2026-05-23 | exp001 | 15.23 | 15.80 | +0.57 | baseline |
| 2026-05-24 | exp002 | 14.81 | 14.95 | +0.14 | anchor delta |
```

目的は、**CVが良いのにLBが悪い実験**と、**LBだけ良くて危ない実験**を見分けること。

---

## 7. 実験ディレクトリ設計

各実験は必ず以下の形式にする。

```text
experiments/
  exp003_cat_typewell/
    config.yaml
    result.json
    cv.csv
    oof.csv
    feature_importance.csv
    train.log
    submission.csv
    notes.md
```

### 7.1 `config.yaml`

実験時に使ったconfigの完全コピー。  
後からconfigを変更しても、実験時点の設定が残るようにする。

### 7.2 `result.json`

機械可読な実験結果。

```json
{
  "exp_id": "exp003_cat_typewell",
  "created_at": "2026-05-23T21:30:00+09:00",
  "model": "catboost",
  "features": ["basic", "anchor", "typewell"],
  "target_mode": "delta_from_anchor",
  "cv_strategy": "GroupKFold_by_well",
  "n_folds": 5,
  "cv_mean": 12.345,
  "cv_std": 0.456,
  "lb_score": null,
  "train_time_min": 18.2,
  "n_features": 128,
  "seed": 42,
  "notes": "typewell hash features added"
}
```

### 7.3 `oof.csv`

OOFは必ず保存する。

最低限の列:

```text
row_id, target, pred, fold, well_id, typewell_id
```

OOFがないと、後で以下ができない。

- error analysis
- model blend
- stacking
- fold別の失敗分析
- well別の誤差分析
- typewell別の誤差分析

### 7.4 `submission.csv`

Kaggle提出用ファイル。  
提出した場合は、LBスコアを `result.json` と `EXP_SUMMARY.md` に追記する。

### 7.5 `notes.md`

人間向けのメモ。

テンプレート:

```markdown
# exp003_cat_typewell

## 目的
typewell hash / duplicate group特徴量が効くか検証する。

## 変更点
- exp002からtypewell.pyの特徴量を追加
- modelはCatBoost
- targetはdelta_from_anchorのまま

## 結果
- CV:
- LB:

## 解釈
- 改善したfold:
- 悪化したfold:
- リーク懸念:

## 次に試すこと
- typewell distance feature
- typewell trend alignment
```

---

## 8. `EXP_SUMMARY.md` の設計

全実験の一覧表。  
Claude/Codexが必ず読むファイル。

```markdown
# Experiment Summary

| Exp | Model | Features | Target | CV | LB | Time | Status | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| exp001 | XGB | basic | direct | 15.23 | 15.80 | 8m | done | cdeotte starter reproduction |
| exp002 | LGB | basic+anchor | delta | 14.81 | 14.95 | 12m | done | anchor improved CV/LB |
| exp003 | CAT | basic+anchor+typewell | delta | 14.62 | - | 18m | pending LB | typewell hash improved CV |
```

Status候補:

```text
planned
running
done
submitted
failed
deprecated
leak_risk
promising
blend_candidate
```

---

## 9. Claude Code / Codex用ルール

### 9.1 `CLAUDE.md`

```markdown
# CLAUDE.md

## Project
ROGII - Wellbore Geology Prediction

## Goal
信頼できるCVをもとに、ROGIIのPublic/Private LBを改善する。

## Non-negotiable Rules
- data/raw は絶対に変更しない
- row random CVは禁止
- 実験は1回につき1目的だけ変更する
- 実験結果は必ず experiments/expXXX/ に保存する
- OOF、submission、result.json、notes.mdを必ず出力する
- EXP_SUMMARY.mdを毎回更新する
- CV改善だけで判断せず、リーク懸念をnotesに書く
- 学習時間は原則20分以内。重い実験は事前に明記する
- 既存の良い実験を壊さない
- feature追加時はablationできるようにfeature set名を分ける

## Preferred Workflow
1. EXP_SUMMARY.mdを読む
2. 直近のbest CV / best LB / leak_risk実験を把握する
3. 新しいexp_idを作る
4. configを作る
5. 実験を実行する
6. result.json / oof.csv / submission.csv / notes.mdを保存する
7. EXP_SUMMARY.mdを更新する
8. 次の改善案を3つだけ提案する

## Important Concepts
- GroupKFold by well_id
- GroupKFold by typewell_id
- last_known_TVT anchor
- typewell duplicate detection
- OOF error slicing
- CV-LB gap
- tree model blend
```

### 9.2 `AGENTS.md`

Codexや他AIエージェント向けの短い指示。

```markdown
# AGENTS.md

You are an experiment assistant for the ROGII Kaggle competition.

Always:
- Create one experiment at a time.
- Never modify raw data.
- Save all artifacts under experiments/expXXX/.
- Update EXP_SUMMARY.md.
- Prefer fast experiments under 20 minutes.
- Do not optimize for public LB only.
- Explain leakage risks.

Before coding:
- Read EXP_SUMMARY.md.
- Check the best previous experiment.
- State the exact hypothesis.

After coding:
- Save result.json.
- Save oof.csv.
- Save submission.csv if applicable.
- Write notes.md.
```

---

## 10. 実行スクリプト

### 10.1 `scripts/run_exp.py`

すべての実験の入口。

実行例:

```bash
python scripts/run_exp.py --config src/rogii/config/exp002_lgb_anchor_delta.yaml
```

処理内容:

1. config読み込み
2. data読み込み
3. fold読み込み
4. features生成
5. CV学習
6. OOF保存
7. test予測
8. submission保存
9. result.json保存
10. EXP_SUMMARY.md更新

### 10.2 `scripts/make_folds.py`

foldを作る。

```bash
python scripts/make_folds.py --strategy group_well --output data/folds/folds_group_well_v001.csv
python scripts/make_folds.py --strategy group_typewell --output data/folds/folds_group_typewell_v001.csv
```

### 10.3 `scripts/analyze_oof.py`

OOFの誤差分析。

```bash
python scripts/analyze_oof.py --exp experiments/exp003_cat_typewell
```

出力:

```text
reports/oof_error_analysis.md
outputs/figures/exp003_oof_by_well.png
outputs/figures/exp003_oof_by_typewell.png
```

### 10.4 `scripts/blend_submissions.py`

blend用。

```bash
python scripts/blend_submissions.py \
  --experiments exp002_lgb_anchor_delta exp003_cat_typewell exp004_xgb_spatial_rolling \
  --method weighted_average \
  --output data/submissions/blend_v001.csv
```

---

## 11. 最初の実験計画

### exp001: XGB Starter再現

目的:

- 公開XGB Starterの再現
- 提出形式の確認
- baseline確保

内容:

```text
Model: XGBoost
Features: basic
Target: direct
CV: starterに合わせる
```

### exp002: Anchor Delta

目的:

- last_known_TVTからの差分予測が効くか検証

内容:

```text
Model: LightGBM
Features: basic + anchor
Target: delta_from_anchor
CV: GroupKFold by well
```

### exp003: Typewell Features

目的:

- Typewellの重複・類似・統計特徴量が効くか検証

内容:

```text
Model: CatBoost
Features: basic + anchor + typewell
Target: delta_from_anchor
CV: GroupKFold by well / typewell
```

### exp004: Spatial + Rolling

目的:

- 井戸軌跡・深度方向の系列特徴量が効くか検証

内容:

```text
Model: XGBoost
Features: basic + anchor + spatial + rolling
Target: delta_from_anchor
CV: GroupKFold by well
```

### exp005: Tree Blend

目的:

- XGB / LGB / CatBoostの多様性をblendで活かす

内容:

```text
Models: exp002 + exp003 + exp004
Method: weighted average / ridge on OOF
```

### exp006: TCN Sequence

目的:

- 深度方向の1D系列モデルがTree系と違う誤差を出すか検証

内容:

```text
Model: TCN / 1D CNN
Input: well-wise depth sequence
Target: direct or delta
GPU: RTX 2080 Super / Colab
```

### exp007: Tree + TCN Blend

目的:

- Tree系と系列NNのblend

内容:

```text
Models: best tree + best TCN
Method: OOF-based blend
```

---

## 12. RTX 2080 Super / Colab運用方針

### ローカルCPUでやること

- EDA
- XGBoost / LightGBM / CatBoost
- 特徴量生成
- CV
- OOF分析
- blend

### RTX 2080 Superでやること

- 小型TCN
- 1D CNN
- 小型Transformer
- NNのseed違い
- PyTorch実験

### Colabでやること

- 長めのTCN/NN
- Colab GPUでの本番学習
- 複数foldのNN学習
- ローカルより重い探索

ただし、ROGIIはGPUで勝つコンペではない。  
まずTree系と特徴量で伸ばし、NNはblend用の多様性モデルとして使う。

---

## 13. 先人の教訓をROGIIに落としたルール

### 13.1 信頼できるCVがない実験は価値が低い

Kaggle上位解法では、ローカル検証の信頼性が繰り返し重視されている。  
ROGIIではrow random CVではなく、井戸・Typewell・空間的な分割を優先する。

### 13.2 OOFを保存しない実験は捨てる

OOFがないと、blend、stacking、誤差分析、fold別比較ができない。

### 13.3 Public LBで一喜一憂しない

Public LBは一部のtestに過ぎない。  
CV-LBの乖離を記録し、リークや過適合を疑う。

### 13.4 Notebookだけで戦わない

Notebookは探索には便利だが、長期戦では破綻する。  
本番処理はsrc/scriptsに移す。

### 13.5 1実験1目的

悪い例:

```text
anchor特徴量、typewell特徴量、モデル変更、fold変更を同時に入れる
```

良い例:

```text
exp002: anchorだけ追加
exp003: typewellだけ追加
exp004: spatialだけ追加
```

### 13.6 Blendは最後ではなく中盤から準備する

OOFを貯めておけば、終盤に強いblendを作れる。  
OOFがないと、最終盤で詰む。

---

## 14. Git / Kaggle / Colab連携

### Git管理するもの

- README.md
- CLAUDE.md
- AGENTS.md
- EXP_SUMMARY.md
- src/
- scripts/
- reports/
- notebooks/の軽量版
- config/

### Git管理しないもの

- data/raw/
- data/processed/
- outputs/
- model weights
- huge OOF
- huge submissions
- cache

### Kaggle Notebookに持ち込むもの

- `src/`
- `scripts/`
- best config
- processed dataset or Kaggle Dataset化したfeature
- 軽量推論コード

### Colabに持ち込むもの

- GitHub repo
- Kaggle API token
- processed dataset
- NN config
- outputをGoogle Driveに保存する仕組み

---

## 15. 最終的な勝ち筋

ROGIIでの勝ち筋は以下。

```text
1. XGB Starterを再現
2. GroupKFoldを設計
3. last_known_TVT anchorを入れる
4. Typewell重複・類似性を使う
5. spatial / rolling / slope特徴量を増やす
6. XGB / LGB / CatBoostを比較
7. OOF誤差を井戸・Typewell・深度位置で切る
8. TCN/1D CNNを補助的に追加
9. OOFベースでblend/stackする
10. CV-LB gapを見て最終submissionを選ぶ
```

---

## 16. 参考リンク

- Cookiecutter Data Science: https://cookiecutter-data-science.drivendata.org/
- Kaggle Grandmasters Playbook: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- Chris Deotte write-ups: https://www.kaggle.com/cdeotte/writeups
- Kaggle Winning Solutions collection: https://www.kaggle.com/code/sudalairajkumar/winning-solutions-of-kaggle-competitions
- Kaggle project management discussion: https://www.kaggle.com/general/4815
- ROGII competition: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
- XGB Starter - CV 15: https://www.kaggle.com/code/cdeotte/xgb-starter-cv-15
