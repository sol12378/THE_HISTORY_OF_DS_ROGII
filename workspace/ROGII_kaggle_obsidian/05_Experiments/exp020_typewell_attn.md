# exp020 — typewell-aware cross-attention NN（offset回収の本命検証）

## 一言
誤差の正体=per-well offset を取りにいく唯一の残路：lateral GR系列と typewell GR-TVT
プロファイルを **cross-attention**（微分可能DTW相当, geom prior近傍へbias）で対応付ける学習モデル。
**結果：NN単体は逆に悪化(16.49)**、typewell照合はノイズ注入に終わり offset を回収できず。
誤差相関0.78と多様化はするため 3-model blend で +0.0198、平滑後 **現best 13.320964**。

## 数値（well-grouped fold）
| | CV |
|---|---:|
| exp020 NN 単体 | 16.489079 |
| exp018 trees | 13.341598 |
| 3-model blend (e18 0.578 / e19 0.364 / e20 0.059) | 13.321789 |
| +exp015平滑 | **13.320964** ← 現best |
- err corr(exp020, exp018)=0.78（exp019の0.895より多様）。alpha(prior bias)≈1.99で初期から殆ど動かず＝GRで prior から外れる根拠を見出せていない。

## 解釈（重要）
- typewellを**正しく cross-attention で投入してもなお** offset を回収できない。
  むしろ単体精度が悪化＝典型的な「ノイズに当てはめて過学習」。
- GR照合の情報限界（offset corr 0.155, [[gr-offset-ceiling]]）が、手作りでも学習モデルでも同じ天井を示す。
- これで GR照合 / 多井戸空間 / 系列NN / typewell-attention の**全paradigmが offset 回収に失敗**。
  → **グローバルCVで CV~5 / LB 6.8 は到達不可能**を最終確認。

## LBについて
public LB=test 3本のみ（分散大）。6.8接近は「グローバル実力」より「3本の引き」要因が大きい可能性。
現best 13.321 を提出して実 CV-LB gap を測るのが次の合理的判断材料。

## 成果物
- code: `scripts/exp020_typewell_attn.py`, `scripts/diag_spatial.py`
- results: `experiments/exp020_typewell_attn/`, `experiments/exp015_seq_smooth_exp020_typewell_attn/`

## リンク
[[exp019_seq_nn]] [[exp018_model_blend]] [[gr-offset-ceiling]] [[Decision_Log]]
