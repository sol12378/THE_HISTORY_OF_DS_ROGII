# exp018 — モデル多様化 blend (LGBM+XGB+CatBoost)

## 一言
exp014幾何特徴量で LGBM/XGBoost/CatBoost を学習し等加重blend。well-fold(leak-free)。
**CV 13.3416**（+0.184 vs exp014）。+exp015平滑化で **13.340426 が新best**（exp015 13.520から+0.180）。

## 数値（well-grouped fold, pre-smoothing）
| model | CV |
|---|---:|
| lgbm | 13.517895 |
| xgb | 13.550757 |
| cat | 13.581831 |
| equal-weight blend | **13.341598** |
| opt-weight blend (0.369/0.305/0.326) | 13.341009 |

→ 頑健性重視で等加重採用。+exp015 mean平滑(w=71) → **13.340426**（全fold一貫+0.0012）。

## 誤差相関
lgbm-xgb 0.960 / lgbm-cat 0.951 / xgb-cat 0.952。高相関だがblendは有効。
同family木モデルの多様化はほぼ限界。これ以上は別系統(系列/NN)か別特徴が必要だが、
per-well offsetが予測不能([[gr-offset-ceiling]] [[diag_self_calib]])のため上積みは限定的。

## 位置づけ
- CV<=5は到達不能（[[gr-offset-ceiling]]）。本実験は**正規手段での現実的最大改善**。
- geom形状が正しく残差はoffset、というoracle知見と整合（modelを変えてもoffsetは消えない）。

## 成果物
- code: `scripts/exp018_model_blend.py`（blend）, `scripts/exp015_seq_smooth.py --model-exp exp018_model_blend`
- results: `experiments/exp018_model_blend/`, `experiments/exp015_seq_smooth_exp018_model_blend/`

## リンク
[[exp014_geom_extrap]] [[exp015_seq_smooth]] [[gr-offset-ceiling]] [[Strategy_2026-05-31]] [[Decision_Log]]
