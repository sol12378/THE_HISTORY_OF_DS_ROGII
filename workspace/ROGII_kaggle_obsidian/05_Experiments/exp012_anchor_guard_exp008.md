# exp012_anchor_guard_exp008 (Track 1)

## 概要
best model exp008 に exp006 流の一方向 anchor guard を後処理適用。閾値+αを OOF sweep。

## 結果
- CV 13.808621 → **13.754461 (+0.054)**
- best: strong_guard(239 wells, α=0.85), mild(15, α=0.5), full_anchor 無効, delta_extreme_thr=50
- test 3 wells は全て no_guard → **公開LB不変**

## Check（重要）
fold別の改善が**非一貫**:
| fold | exp008 | guard | delta |
|---|---:|---:|---:|
| 0 | 13.0397 | 13.1170 | **-0.0773** |
| 1 | 14.0416 | 13.8562 | +0.1853 |
| 2 | 13.5341 | 13.5064 | +0.0277 |
| 3 | 13.9626 | 13.8653 | +0.0973 |
| 4 | 14.4256 | 14.3957 | +0.0299 |

fold0 は悪化、fold1 が改善を牽引。864通り(full_hidden×delta×α_strong×α_mild)を
pooled OOF で sweep → OOF過学習の兆候。

## 判定
低信頼の小改善。**本採用は保留**。strong_guard α=0.85 の頑健部分のみ将来 nested-CV で再評価。
leak懸念: guard条件は test計算可能特徴量のみ→ leak-free。

## リンク
[[exp006_anchor_guard]] [[exp008_gr_rolling]] [[Decision_Log]] [[Strategy_2026-05-31]]
