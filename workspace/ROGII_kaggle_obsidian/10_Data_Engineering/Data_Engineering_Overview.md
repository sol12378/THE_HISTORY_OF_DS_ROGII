# データエンジニアリング概要

## 目的

raw Kaggle files から model-ready feature tables まで、再現可能で監査しやすいパイプラインを作る。

このパイプラインが満たすべきこと:

- 実験を再現できる
- リークを防げる
- train/testで一貫した特徴量を作れる
- 信頼できるCV foldを作れる
- OOFとsubmission artifactを保存できる
- well単位の失敗分析ができる

## 中核原則

各wellはまずsequenceとして扱い、その後tabular rowsとして扱う。

raw filesはwell単位で整理されている。モデルは行単位で学習することが多いが、特徴量生成では以下のwell-level contextを保持する。

- row order
- Prediction Start position
- pre-PS anchor state
- rolling windows within a well
- typewell reference curve

## Data Layers

推奨レイヤー:

```text
data/raw/
  Kaggleから取得した不変ファイル。

data/interim/
  parse済み・軽く正規化したwell単位テーブル。

data/processed/
  model-readyなbase tableとfeature table。

data/folds/
  fold assignment files。

outputs/
  models, OOF, predictions, plots, diagnostics。

experiments/
  実験ごとの凍結configと結果。
```

## Pipeline Stages

1. Raw inventory
2. Schema validation
3. Horizontal well parsing
4. Typewell parsing
5. Prediction Start detection
6. Base table construction
7. Fold generation
8. Feature generation
9. Model dataset export
10. OOF/submission artifact validation

## 変更しないルール

- `data/raw/` は絶対に変更しない。
- test-time featureを作るときにfuture `TVT` を使わない。
- row-random splitはしない。
- `well_id`, `row_idx`, 必要に応じてsubmission `id` を必ず保存する。
- どの予測もsource CSV rowへ辿れるmetadataを残す。

## 固定する設計判断

1. `well_id` と `row_idx` をcanonical row keysとする。
2. 学習行は horizontal-well rows のうち `is_target=True` の行とする。
3. ローカル評価は `is_target=True` に対する `TVT` RMSEとする。
4. foldsは一度作ったら `data/folds/` でversion管理する。
5. base tableは軽量で安定させ、実験固有の特徴量はfeature tableまたはfeature functionに置く。
6. processed tableにはversion、作成時刻、raw data fingerprint、schema contractを持たせる。
7. feature familyは独立にablationできるよう加算的に設計する。
8. train-only geological marker columnsは、test parityを明示しない限りanalysis-onlyとする。

## 推奨データアーキテクチャ

bronze / silver / gold の考え方で分ける。

```text
Bronze:
  data/raw/
  Kaggle filesそのもの。不変。

Silver:
  data/interim/
  stable row IDs、source paths、basic validationを持つ正規化済みテーブル。

Gold:
  data/processed/
  model-ready base tables、fold files、feature tables。
```

実験のデフォルト入力はGold layerだけにする。

## テーブル粒度

canonical granularityは3つ。

```text
well-row:
  horizontal well pointごとに1行
  key = split, well_id, row_idx

typewell-row:
  typewell TVT pointごとに1行
  key = split, well_id, typewell_row_idx

well-level:
  wellごとに1行
  key = split, well_id
```

明示的なjoin ruleなしに粒度を混ぜない。

## Metadata Requirements

生成datasetごとにsidecar metadata fileを持たせる。

```text
data/processed/train_base_v001.parquet
data/processed/train_base_v001.meta.json
```

metadataに含めるもの:

- dataset name
- version
- created_at
- source file counts
- source raw fingerprint
- row counts
- column list
- null summary
- script name
- git commit if available
- leakage assumptions
- validation checks passed

## Change Policy

base schemaを変える必要がある場合:

1. `train_base_v002.parquet` のような新versionを作る。
2. 依存実験がarchiveされるまで旧versionを残す。
3. 変更理由を `Decision_Log.md` に書く。
4. `Schema_and_Contracts.md` を更新する。
5. 過去実験が使ったbase tableを黙って上書きしない。
