# exp019 — 系列NN (per-well TCN) + exp018 ensemble

## 一言
ユーザー要望「NN modelを加えたensembleでCVを5台に」。well全体のGR+幾何 系列を
dilated 1D CNN(TCN)で学習(geom priorへのresidual)。**NN単体14.63 / blend 13.332 / +平滑 13.331**。
**CV~5には到達せず**。系列NNでも per-well offset を復元できないことを確認(情報限界)。

## 数値（well-grouped fold）
| | CV |
|---|---:|
| NN単体 (TCN) | 14.630049 |
| exp018 (LGBM+XGB+Cat) | 13.341598 |
| blend (exp018 0.922 / NN 0.078) | 13.332072 |
| +exp015平滑(mean w=71) | **13.331211** ← 現best |

- 誤差相関 corr(NN, exp018) = **0.8955**（NNは同じ幾何信号を学習、独立情報ほぼ無し）。
- blend最適加重がNNに0.078しか割かない=NNの独立寄与が小さい。

## なぜ5に届かないか（既出の情報限界）
[[gr-offset-ceiling]]: geom形状は正しく残差は低周波offset。理想offsetなら seg_oracle=4.4。
だが offset を当てる信号が無い: GR corr=0.155、self-calib corr=0.0([[diag_self_calib相当]])。
系列NN(GR系列形状を端から端まで見る唯一の paradigm)でも offset 復元不可・誤差相関0.895。
→ tree/NN/GR照合の全手段で **CV~5 は正規手段で到達不可能**。LB首位でも6.8。

## 設計メモ
- 各well を L=1024 グリッドへリサンプル(offsetは低周波なので十分)。入力11ch(GR正規化/欠損/known/known_delta/幾何/geom prior)。
- pred_delta = geom_prior + TCN(resid)。masked MSE(hiddenのみ)。folds_group_well_v001。
- leak-free: typewell不使用、hidden TVT不使用、well-grouped fold。

## 成果物
- code: `scripts/exp019_seq_nn.py`
- results: `experiments/exp019_seq_nn/`, `experiments/exp015_seq_smooth_exp019_seq_nn/`

## リンク
[[exp018_model_blend]] [[gr-offset-ceiling]] [[exp014_geom_extrap]] [[Decision_Log]]
