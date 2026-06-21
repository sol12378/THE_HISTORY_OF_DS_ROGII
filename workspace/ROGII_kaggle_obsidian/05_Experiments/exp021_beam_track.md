# exp021_beam_track — Beam Search GR-typewellトラッカー

## 概要
参照の beam_search(14 config, DTW型)を忠実移植、773-well leak-free CV。
typewell index を1行±2ステップ動かし cost=GR誤差²/es+移動コスト の累積最小経路を bs本ビーム探索。

## 結果
| 手法 | CV |
|---|---|
| anchor | 15.910 |
| **beam (faithful)** | **15.697** |
| beam (+offset較正) | 15.752 |

anchorに勝つ 432/773。3 test: 8.38/14.78/8.49（2/3でanchorに負け）。

## 判定: 失敗（≈anchor）
- ±2運動制約が粗く、長距離hiddenで累積誤差・迷子。決定論ビームは一度ずれると復帰しにくい。
- 同じGR-typewell情報でも Particle Filter(exp022)は CV 11.02・3test平均≈5.0 と大きく勝つ。
  → トラッキングは**確率的(PF)が決定論的(Beam)より優位**。Beamは不採用。

## リンク
[[exp022_particle_filter]] [[exp023_leak_lookup]]
