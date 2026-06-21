# exp033_final_blend — 全component最終blend (現best CV 9.977)

## 概要
exp022(原PF)+exp025(tuned PF)+exp030b(multi-tw PF)+exp031(physical-lik PF)+exp014(geom)+exp018(trees)+exp019(NN)+exp020(attn)
の8 components を nested 5-fold NNLS でblend、平滑(w=101)。

完全leak-free。TabICL (exp032b) は時間/メモリ制約で blend未組み込み (将来課題)。

## 結果
| 段階 | CV |
|---|---:|
| 前best (exp026) | 10.062 |
| NNLS nested 8-blend | 9.989 |
| **+ 平滑(w=101)** | **9.977** |

**改善 −0.085**。

## 平均NNLS重み (over folds)
| Component | Weight | 単体CV |
|---|---:|---:|
| pf_orig (exp022) | **0.308** | 11.024 |
| attn (exp020) | **0.285** | 13.322 |
| pf_tuned (exp025) | **0.214** | 10.984 |
| pf_multi (exp030b) | 0.123 | 12.146 |
| trees (exp018) | 0.121 | 13.342 |
| pf_phys (exp031) | 0.057 | 20.379 |
| geom (exp014) | 0.017 | 13.525 |
| nn (exp019) | 0.000 | 13.332 |

主軸: **pf_orig + pf_tuned + attn (合計0.81)**。残りは多様化のための小寄与。
geom/trees/nn/attnは相関0.985-0.999で実質1シグナル → NNLSがattnを代表として採用。

## 誤差相関 (delta-space)
- pf系内部: 0.81-0.95 (互いに強相関)
- pf vs tree系: 0.37-0.44 (中程度の相補)
- pf_phys vs others: **0.29-0.35** (最も独立) ← 物理尤度がもたらす多様化

## test wells (vs train真値, 3wells)
- 000d7d20: 3.99 (PF 3.99)
- 00bbac68: 4.62 (PF 4.44)
- 00e12e8b: 5.64 (PF 6.52)
- 平均 ≈ 4.75 (PF単独平均5.0から改善)

## 予想LB
CV→LB gap安定~1.4 (exp003/008/026で再現) → 予想 LB **8.4〜8.6**
- exp026 LB 8.672 ⇒ 推定 LB 8.5±0.2

## 残された天井
- per-well offsetはleak-free信号で予測不能 (corr=0.155 GR/-0.028 self-calib)
- 47壊れwellの大半は依然解けない (4/47救出のみ)
- 上位LB6.5は本来leak-free信号外（公開不可信号、ハイパーチューニング、TabICL大規模化など）

## 次の打ち手
1. **TabICL** をGPU/Kaggle kernel上で実行し追加(±-0.1〜-0.3 期待)
2. **Multi-typewell GR-similarity選定**でexp030b強化(broken well救出狙い)
3. **PF統合likelihood** (raw + normalized + derivative の同一PF内mix) — exp031を発展

## リンク
[[exp026_final_blend]] [[exp030b_multi_tw_vec]] [[exp031_pf_physical_lik]]
