# Raw to Processed Pipeline

## 入力

raw input paths:

```text
data/raw/train/{well_id}__horizontal_well.csv
data/raw/train/{well_id}__typewell.csv
data/raw/train/{well_id}.png
data/raw/test/{well_id}__horizontal_well.csv
data/raw/test/{well_id}__typewell.csv
data/raw/sample_submission.csv
```

## 出力

主要なprocessed files:

```text
data/processed/train_base_v001.parquet
data/processed/test_base_v001.parquet
data/processed/typewell_train_base_v001.parquet
data/processed/typewell_test_base_v001.parquet
data/folds/folds_group_well_v001.csv
```

任意のfeature tables:

```text
data/processed/features_anchor.parquet
data/processed/features_trajectory.parquet
data/processed/features_gr.parquet
data/processed/features_typewell.parquet
```

feature tablesはversion付きにする。

```text
data/processed/features_anchor_v001.parquet
data/processed/features_trajectory_v001.parquet
data/processed/features_gr_v001.parquet
data/processed/features_typewell_basic_v001.parquet
data/processed/features_alignment_v001.parquet
```

## Train Base Table

目標schema:

```text
split
well_id
row_idx
MD
X
Y
Z
GR
TVT_input
TVT
is_target
ps_idx
last_known_TVT
last_known_MD
last_known_X
last_known_Y
last_known_Z
delta_MD_from_PS
delta_X_from_PS
delta_Y_from_PS
delta_Z_from_PS
```

追加推奨の安定列:

```text
source_path
n_rows_in_well
hidden_length
known_length
is_known_tvt
is_gr_missing
md_from_start
md_to_end
row_frac
post_ps_step
```

## Test Base Table

目標schema:

```text
split
well_id
row_idx
id
MD
X
Y
Z
GR
TVT_input
is_target
ps_idx
last_known_TVT
last_known_MD
last_known_X
last_known_Y
last_known_Z
delta_MD_from_PS
delta_X_from_PS
delta_Y_from_PS
delta_Z_from_PS
```

追加推奨の安定列:

```text
source_path
n_rows_in_well
hidden_length
known_length
is_known_tvt
is_gr_missing
md_from_start
md_to_end
row_frac
post_ps_step
```

## なぜBase Tableが重要か

すべての実験は同じbase tableから始める。これにより以下の隠れた差分を防ぐ。

- Prediction Start detection
- row indexing
- train/test feature parity
- fold assignment
- target masking
- submission row mapping

## Build Order

推奨build order:

1. `raw_inventory_v001.json`
2. `horizontal_train_interim_v001.parquet`
3. `horizontal_test_interim_v001.parquet`
4. `typewell_train_interim_v001.parquet`
5. `typewell_test_interim_v001.parquet`
6. `train_base_v001.parquet`
7. `test_base_v001.parquet`
8. `folds_group_well_v001.csv`
9. feature family tables
10. experiment matrices

## 実装済みコマンド

base table生成:

```bash
.venv/bin/python scripts/build_base_tables.py
```

fold生成:

```bash
.venv/bin/python scripts/make_folds.py
```

validation:

```bash
.venv/bin/python scripts/validate_data.py
```

## Submission Mapping Contract

testでは以下を保証する。

```text
sample_submission.id == test_base[test_base.is_target].id
```

これはすべてのsubmission前に検証する。
