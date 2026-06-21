# exp001_anchor_baseline

## 仮説

PS直前の `last_known_TVT` をtarget区間にそのまま延長すると、モデルが最低限超えるべき基準線になる。

## 実装

- script: `scripts/run_baseline_anchor.py`
- 入力: `data/processed/train_base_v001.parquet`, `data/processed/test_base_v001.parquet`
- fold: `data/folds/folds_group_well_v001.csv`
- 予測: `pred_tvt = last_known_TVT`
- 評価対象: `is_target=True` のtrain rows

## 結果

| metric | value |
|---|---:|
| overall CV RMSE | 15.909853 |
| fold mean RMSE | 15.887139 |
| fold std RMSE | 0.854557 |
| OOF rows | 3,783,989 |
| submission rows | 14,151 |

fold別RMSE:

| fold | RMSE |
|---:|---:|
| 0 | 15.026314 |
| 1 | 15.926525 |
| 2 | 14.943437 |
| 3 | 16.289956 |
| 4 | 17.249465 |

## 解釈

かなり強い単純baseline。`last_known_TVT` からtarget区間の真値が大きく逸脱しないwellが多い一方、fold 4 のように難しい分割もある。以後のモデルは、anchorからの差分 `TVT - last_known_TVT` をどれだけ安定して補正できるかを見る。

## リーク懸念

`last_known_TVT` はtarget開始前の既知TVTから作っているため、基本的なリーク懸念は低い。ただしbase table生成時にPS以後の真値TVTを参照していないことを継続して検証する。

## 次アクション

[[exp003_lgb_anchor_trajectory]] でanchor差分をLightGBMに学習させ、改善量を見る。
