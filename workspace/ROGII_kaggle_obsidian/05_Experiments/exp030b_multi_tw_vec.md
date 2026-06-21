# exp030b_multi_tw_vec — Multi-typewell PF (vectorized)

## 概要
地質資料§3に基づき、各wellに対しK=3のtypewell候補で並走PF→尤度加重平均。
ROGII公式StarSteer/GeoAssistの「複数offset wellから最良相関選択」設計を確率フィルタ化。

候補選定: own typewell + 空間最近傍 distinct typewell ×2 (signature重複除去)。
vectorized PF (S=32 seeds × N=500 particles × K=3 cands)、ProcessPoolExecutor 8 workers。

## 結果
| 指標 | exp030b | exp022 (own only) |
|---|---:|---:|
| pooled CV | 12.146 | 11.024 |
| beat anchor | 573/773 | 587/773 |
| broken(>20) | 53 | 47 |
| best_cand=own | 293 (38%) | n/a |
| best_cand=alt1 | 227 (29%) | |
| best_cand=alt2 | 253 (33%) | |

vs exp022: better=326, worse=447, mean delta=-0.35 (悪化)。
exp022の47壊れwell中、新PFで復活(<=20): **4/47**(救出弱い)。

## 解釈
- 空間最近傍だけのtypewell選定では地質ブロックが異なる候補が混入し悪化
- 62%のwellで「自分以外」が高尤度に見える = S=32の尤度推定にノイズ
- ただし**誤差相関 0.81 vs exp022**（強相関）→単体は同じ系列の縮小版
- 地質資料§3.3の「GR範囲似+構造ブロック同じ」基準が抜けている

## ブレンド寄与
exp033 final blendで weight 0.123 (PFサイズの2/3)。直交性は限定的だが微小寄与あり。

## 次の改善案
- GR-similarity-basedの候補選定 (build区間GRパターン類似度)
- own typewellに強prior bias、only switch if alternative log-lik >>
- S=32→128へ戻し尤度推定の安定性向上

## リンク
[[exp022_particle_filter]] [[exp031_pf_physical_lik]] [[exp033_final_blend]]
