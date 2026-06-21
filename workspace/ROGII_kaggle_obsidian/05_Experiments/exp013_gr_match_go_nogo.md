# exp013_gr_match_go_nogo (Track 2 Phase 1)

## 概要
GR-typewellアライメント本丸仮説の GO/NO-GO 検証（第1弾）。
hidden観測GRを typewell GR-TVTに照合しTVT直接復元。**完全leak-free**（ラベル不使用・fold不要）。

## 手法
anchor±110窓のTVTグリッド(0.5刻み)から |tw_GR(tvt) - (obs_GR - offset)| 最小TVTを探索。
variantA(nearest) / variantB(連続性λ=0.15) / variantA_smooth(rolling median 25)。

## 結果（全target行 RMSE）
| method | RMSE |
|---|---:|
| anchor | 15.91 |
| exp008(参照) | 13.81 |
| variantA | **43.08** |
| variantB | 46.23 |
| variantA_smooth | 24.16 |

corr>0.85帯(244 wells)でも A≈41.5。B が anchor に勝つ well: 15/773。

## 解釈
素朴GR直接照合は**全滅**。GRが窓内で多価（高周波変動 std≈26）→ 遠方偽マッチを拾い誤差爆発。
hidden TVTのdelta幅±100だが中央値0.84で大半anchor近傍なので、広窓照合は害。
→ 狭窓+幾何prior中心の追試 [[exp013b_gr_local_refine]] へ。

## リンク
[[exp013b_gr_local_refine]] [[exp009b_typewell_gr_well_only]] [[Strategy_2026-05-31]] [[group-e-sealed]]
