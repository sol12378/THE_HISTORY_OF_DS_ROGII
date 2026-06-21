# Schema and Contracts

## File-Level Contracts

各horizontal well fileは以下を満たす。

- has `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`
- train additionally has `TVT`
- `MD` is monotonic increasing
- `row_idx` is the original zero-based CSV row index
- `TVT_input` missing region should be a contiguous tail

各typewell fileは以下を満たす。

- has `TVT`
- has `GR`
- train may have `Geology`
- `TVT` should be sorted or sortable

## Identifier Contracts

canonical IDs:

```text
well_id = filename before "__"
row_idx = original integer row position within horizontal CSV
submission_id = f"{well_id}_{row_idx}"
```

processed dataから `well_id` と `row_idx` を落とさない。

## Naming Contracts

raw column namesは大文字のまま使う。

```text
MD, X, Y, Z, GR, TVT, TVT_input
```

engineered columnsはsnake_caseを使う。

```text
well_id
row_idx
ps_idx
last_known_TVT
delta_MD_from_PS
is_target
```

scriptごとにraw columnsの名前を変えない。

## Target Contract

training target rows:

```text
is_target = TVT_input.isna()
```

evaluationは `is_target=True` の行だけを使う。

## Feature Parity Contract

modelに使うfeatureは、test target `TVT` を見ずにtrain/test両方で計算できなければならない。

train-only columns:

- `TVT`
- `ANCC`
- `ASTNU`
- `ASTNL`
- `EGFDU`
- `EGFDL`
- `BUDA`
- train typewell `Geology`

train-only columnsは、リークリスクを明記したうえで分析や補助target用途に限定して使う。

## Base Schema v001

最初のstable base schemaを `v001` とする。

必須well-row columns:

```text
split
well_id
row_idx
source_path
MD
X
Y
Z
GR
TVT_input
TVT                  # train only, absent or null in test
id                   # test submission id, optional/null in train
is_target
is_known_tvt
is_gr_missing
ps_idx
n_rows_in_well
known_length
hidden_length
last_known_TVT
last_known_MD
last_known_X
last_known_Y
last_known_Z
delta_MD_from_PS
delta_X_from_PS
delta_Y_from_PS
delta_Z_from_PS
post_ps_step
row_frac
```

任意のanalysis-only columns:

```text
ANCC
ASTNU
ASTNL
EGFDU
EGFDL
BUDA
```

base tableに含める場合、analysis-onlyであることを明示し、default model feature listから除外する。

## Validation Gates

以下を満たさないprocessed tableはvalidとみなさない。

- all required columns exist
- row count matches raw source row count
- `well_id`, `row_idx` is unique within split
- `ps_idx` is not null for every well
- `hidden_length > 0`
- train `TVT` is non-null
- test `id` exists for all target rows
- sample submission target IDs match test target IDs exactly
