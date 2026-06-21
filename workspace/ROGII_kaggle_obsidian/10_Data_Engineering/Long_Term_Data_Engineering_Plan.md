# 長期データエンジニアリング計画

## 目的

本格的なmodelingに入る前に、データエンジニアリング構成を固定する。目的は、コンペ中盤以降にschemaが揺れてCV比較が壊れることを防ぎ、すべての実験を再現可能にすることである。

## アーキテクチャ概要

以下の4層を安定したレイヤーとして使う。

```text
raw
  不変のKaggle files

interim
  parse済み・正規化済みテーブル

processed
  version管理されたbase tableとfeature table

experiment
  exact matrices, OOF, submissions, model artifacts
```

## 安定成果物

以下の成果物はstable contractとして扱う。

```text
data/processed/train_base_v001.parquet
data/processed/test_base_v001.parquet
data/processed/typewell_train_base_v001.parquet
data/processed/typewell_test_base_v001.parquet
data/folds/folds_group_well_v001.csv
```

Feature artifactsは独立にversion管理する。

```text
data/processed/features_anchor_v001.parquet
data/processed/features_trajectory_v001.parquet
data/processed/features_gr_v001.parquet
data/processed/features_typewell_basic_v001.parquet
data/processed/features_alignment_v001.parquet
```

## 固定すべきもの

固定するもの:

- canonical identifiers
- base schema
- target row definition
- fold file format
- submission mapping
- feature family boundaries
- artifact naming conventions
- leakage policy

固定しないもの:

- model type
- feature versions
- target transform choice
- blending strategy
- experiment configs

## Canonical Identifiers

すべてのrow-level tableは以下を持つ。

```text
split
well_id
row_idx
```

test target rowsはさらに以下を持つ。

```text
id = f"{well_id}_{row_idx}"
```

これらのidentifierは、join、OOF analysis、error slicing、submission generationの背骨である。

## Base Tables vs Feature Tables

Base tableには、安定していて全実験で使う列だけを入れる。

- raw observable fields
- row identity
- Prediction Start metadata
- anchor metadata
- target and mask columns

Feature tableには、実験で変わりうる派生列を入れる。

- rolling GR
- trajectory curvature
- typewell joins
- alignment scores
- candidate TVT estimates

この分離により、実験入力の安定性を保ちながら、特徴量ロジックを安全に変えられる。

## Fold Strategy

default fold contract:

```text
data/folds/folds_group_well_v001.csv
columns: split, well_id, fold
```

row-level trainingでは、`well_id` でfoldをjoinする。

追加のfold fileを作る場合:

```text
folds_group_typewell_v001.csv
folds_spatial_cluster_v001.csv
folds_adversarial_public_v001.csv
```

実験が依存したfold fileは上書きしない。

## Quality Gates

ETL runは以下に該当したら即失敗させる。

- raw file counts do not match expectations
- sample submission IDs do not map to test target rows
- `TVT_input` missing section is not a contiguous tail
- row keys are duplicated
- train/test feature columns diverge
- any target-aware feature is generated before fold isolation
- any experiment tries to use analysis-only columns by default

## 推奨実装モジュール

```text
src/rogii/data/raw_inventory.py
src/rogii/data/build_base.py
src/rogii/data/build_typewell.py
src/rogii/data/validate_schema.py
src/rogii/data/make_folds.py
src/rogii/features/build_anchor.py
src/rogii/features/build_trajectory.py
src/rogii/features/build_gr.py
src/rogii/features/build_typewell.py
src/rogii/features/build_alignment.py
src/rogii/training/build_matrix.py
```

## 推奨スクリプト

```text
scripts/build_base_tables.py
scripts/build_feature_table.py
scripts/make_folds.py
scripts/validate_data.py
scripts/build_matrix.py
scripts/run_exp.py
```

## Public Repository Safety

commitしないもの:

- `.env`
- `.kaggle/`
- raw Kaggle data
- processed parquet files
- OOF files
- model artifacts
- submissions
- Obsidian workspace files

commitしてよいもの:

- ETL code
- schema docs
- small templates
- configs
- empty `.gitkeep` directories
- Obsidian Markdown notes

## Migration Policy

設計変更が必要な場合:

1. create a new artifact version
2. leave old artifacts untouched
3. document the reason in `Decision_Log.md`
4. update relevant schema docs
5. list impacted experiments
6. do not compare CV across incompatible data versions without noting it

## 最終原則

base data layerは退屈なくらい安定し、追跡可能であるべき。新しい工夫はfeature versionsとexperimentsで行い、row identity、masks、folds、target definitionsを黙って変えてはいけない。
