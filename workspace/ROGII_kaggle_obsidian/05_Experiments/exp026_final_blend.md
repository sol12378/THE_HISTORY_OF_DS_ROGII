# exp026_final_blend — tuned PF 最終再ブレンド（現best 10.062）

## 概要
exp024 と同じ NNLS nested-fold ブレンド+平滑だが、PFソースを **tuned PF(exp025_pf_tuned, init_spread=4,PN=0.01)** に差替えた最終版。

## 結果
| 段階 | CV |
|---|---|
| tuned PF 単体(exp025b) | 10.984 |
| NNLS nested-fold blend | 10.076 |
| **+ 平滑(w101)** | **10.062** |

NNLS重み: **pf 0.670 + attn 0.471**(geom/trees/nn≈0)。fold一貫: 9.42/10.03/9.65/10.02/11.10。

## 解釈
- exp024(untuned PF)の10.077から **−0.015のみ**。PFチューニングのfull寄与は僅少
  (subsetの−0.258は非代表＝subset過学習だった)。
- **改善の99%はPF統合そのもの**(前best13.32→10.08)。チューニングは誤差範囲。
- 最終best = **10.062**、通算 13.32→10.062 = **−3.26**。完全leak-free・nested-fold honest。

## 今後
- これ以上のCV低減はPF自体の質向上(別の追跡モデル/特徴)か、47壊れwellの扱いが必要だが
  reliabilityがleak-free予測不能のため頭打ち感。
- 次の現実的一手: **Kaggle kernel化してLB転移を検証**(3 test wellでPF≈5.0→LB一桁圏が射程)。要許可。

## リンク
[[exp024_multistage_blend]] [[exp025_pf_tune]] [[exp022_particle_filter]]
