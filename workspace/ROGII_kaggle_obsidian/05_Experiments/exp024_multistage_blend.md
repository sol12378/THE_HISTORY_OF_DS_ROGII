# exp024_multistage_blend — PF×geom×trees/NN 多段ブレンド（現best 10.077）

## 概要
PF(exp022, CV11.02)を既存の幾何系ensemble(geom/trees/nn/attn)とNNLSブレンド。
delta スケール(pred−anchor)で非負最小二乗、**nested 5-fold で正直CV**、+well内mean平滑(w101)。

## 結果
| 段階 | CV |
|---|---|
| anchor | 15.910 |
| 個別: pf | 11.024 |
| 個別: geom/trees/nn/attn | 13.53/13.34/13.33/13.32 |
| **NNLS nested-fold blend** | **10.090** |
| + 平滑(w101) | **10.077** |
| full-data weights(optimistic) | 10.059 |

前best(exp020 blend) 13.321 から **+3.24改善**。fold一貫: 9.28/9.97/9.91/10.05/11.08。

## 誤差相関とNNLS重み
- pf vs 幾何系: **0.43-0.44**(直交) / 幾何系どうし: 0.98-1.00(冗長)。
- NNLS full重み: **pf 0.665 + attn 0.467**、geom/trees/nn ≈ 0（attnが最良幾何代表として吸収）。
- 重みはfold間で安定(pf 0.66±0.01, attn 0.36-0.49)→過学習低。

## 解釈
- PFが直交信号、幾何ensembleが補完。**ブレンドの主役はPF**(重み0.66)。
- 幾何系4本は相関ほぼ1で1本分の情報。NNLSが自動的にattn1本に集約。
- nested CV(10.09)≈full(10.06)で過学習なし。

## 壊れwellゲートは不成立(別途検証)
47壊れwell(pf_rmse>20)のオラクル上限(→attn)=9.46だが、leak-free検出シグナル(PF自己GR残差,
corr0.225)では分離不能。gate適用はむしろ悪化。PF信頼性は予測不能→NNLS重みが最善の頑健化。

## 次
- PFパラメータ調整(exp025)で PF自体を強化→再ブレンド。
- Kaggle kernel化してLB転移検証(要許可)。3 test wellでPF≈5.0なのでLB一桁圏が射程。

## リンク
[[exp022_particle_filter]] [[exp014_geom_extrap]] [[exp020_typewell_attn]] [[exp025_pf_tune]]
