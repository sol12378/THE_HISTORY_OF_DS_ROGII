# Feature Table Design

## 設計

stable keysを持つ加算型のfeature familyを使う。

```text
key: split, well_id, row_idx
```

各feature familyは独立にbuildできるようにする。

```text
anchor_features
trajectory_features
gr_features
typewell_features
alignment_features
```

これによりablationがしやすくなる。

```text
basic
basic + anchor
basic + anchor + trajectory
basic + anchor + trajectory + GR
basic + anchor + trajectory + GR + typewell
```

## Anchor Features

Prediction Start以前の行だけを使い、wellごとに計算する。

- last known TVT
- last known MD/X/Y/Z
- delta from PS
- local TVT slope before PS
- local TVT curvature before PS

## Trajectory Features

現在位置と近傍位置から計算する。

- local direction
- displacement from PS
- horizontal distance
- dZ/dMD
- azimuth proxy
- inclination proxy
- curvature

## GR Features

well内で計算する。

- raw GR
- missing flag
- interpolated value
- rolling mean/std
- rolling diff/slope
- GR percentile
- missing run length

## Typewell Features

対応するtypewellから計算する。

- GR stats over full typewell
- GR at last known TVT
- GR at baseline predicted TVT
- local rolling stats around candidate TVT

## Alignment Features

horizontal GR window と typewell GR window を比較して計算する。

- best correlation score
- best matching TVT
- shift from baseline TVT
- DTW-like distance
- confidence from GR coverage

## Storage

parquetを推奨する理由:

- faster load
- preserves dtypes
- compact
- stable for repeated experiments

## Feature Registry

軽量なfeature registryを維持する。

```text
data/processed/feature_registry_v001.json
```

各feature familyについて記録するもの:

- feature table path
- version
- keys
- columns
- dependencies
- leakage assumptions
- train/test availability
- generating script/function

## Feature Stability Rule

一度実験で使ったfeature tableは上書きしない。

代わりにversionを上げる。

```text
features_gr_v001.parquet
features_gr_v002.parquet
```

experimentsはconfig内で正確なfeature versionを参照する。

## Matrix Assembly

training matrixは以下から組み立てる。

```text
base table
+ selected feature family versions
+ fold file
+ target transform
```

生成されたmatrixはcacheしてよい。

```text
data/cache/matrix_exp003_lgb_anchor_trajectory.parquet
```

cache filesは捨ててもよい。base tablesとversioned feature tablesは捨てない。

## Target Transform Contract

supported target modesは明示する。

```text
direct:
  y = TVT

delta_from_anchor:
  y = TVT - last_known_TVT

slope_from_anchor:
  y = (TVT - last_known_TVT) / max(delta_MD_from_PS, eps)
```

evaluationでは必ずabsolute `TVT` に戻してRMSEを計算する。

## Train/Test Feature Parity

training前に以下を検証する。

```text
set(train_features) == set(test_features)
```

例外は必ず文書化し、codeで明示的に扱う。黙って無視しない。
