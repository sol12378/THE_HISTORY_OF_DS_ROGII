# 再現性とArtifacts

## 実験ごとの必須Artifacts

```text
experiments/expXXX/
  config.yaml
  result.json
  cv.csv
  oof.csv
  feature_importance.csv
  train.log
  submission.csv
  notes.md
```

## Result JSON

含めるべきもの:

- exp_id
- created_at
- model
- features
- target mode
- CV strategy
- fold scores
- CV mean/std
- LB score if submitted
- seed
- data version
- code version or git commit when available

## Data Versioning

raw dataは不変なので、processed dataはfilenameまたはmetadataでversion管理する。

```text
train_base_v001.parquet
features_anchor_v001.parquet
folds_group_well_v001.csv
```

schema changeはObsidianとexperiment notesに必ず記録する。
