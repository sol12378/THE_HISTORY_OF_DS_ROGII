# exp013b_gr_local_refine (Track 2 Phase 1 確定)

## 概要
exp013(広窓全滅)の追試。**狭窓+幾何prior中心**ならGRが局所微修正できるか?
center∈{anchor, exp008} の±w窓(10/20/30)でGR照合、centerに最近い解採用。完全leak-free。

## 結果（全target行 RMSE、抜粋）
| method | RMSE | vs exp008 |
|---|---:|---:|
| exp008(参照) | **13.808621** | BEST |
| blend_exp008_w10_bw0.3 | 13.825925 | +0.017 |
| matchE_exp008_w10_smooth | 13.905225 | +0.097 |
| matchA_exp008_w10 (純) | 14.407844 | +0.599 |
| anchor | 15.909853 | +2.10 |
| matchA_anchor_w30 | 18.771290 | +4.96 |

## 判定: **NO-GO（GR-typewell直接照合）**
- どのGR変種も exp008 を超えない。最善でも +0.017悪化（実質exp008）。
- 純match は±10窓でも +0.6〜2.6悪化。anchor中心は全てanchor以下。
- **GRは強prior上で上乗せ情報ゼロ**。

### なぜ
1. GR↔TVT多価で±10窓ですら逆変換不安定。
2. exp008 Group D(GR rolling統計)が既にGR信号を吸収。
3. anchor/幾何が既に強い。

## 含意
exp009→exp013→exp013b 一貫で失敗 → 「GRアライメント=本丸」棄却。
Tier2投資はモデル多様化・系列性・幾何特徴へ振替（[[Strategy_2026-05-31]]）。
残路はDTW系列照合のみ（高リスク・後回し）。

## リンク
[[exp013_gr_match_go_nogo]] [[Strategy_2026-05-31]] [[Decision_Log]]
