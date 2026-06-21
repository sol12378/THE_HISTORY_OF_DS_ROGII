# Submission 002 — exp026 PF×geom 自己完結blend

## 提出情報
| 項目 | 値 |
|---|---|
| 日時 | 2026-06-03 |
| 手法 | PF(GR-typewell系列トラッキング) × geom(幾何外挿LightGBM) NNLSブレンド+平滑 |
| local CV | 10.06 (nested-fold, leak-free) |
| **Public LB** | **8.672** |
| 順位 | **~63 / 2145（上位2.9%）** |
| kernel | sol12378/rogii-exp026-pf-geom-blend |

## CV-LB転移（最重要）
| exp | CV | LB | gap(CV−LB) |
|---|---|---|---|
| exp003 | 15.05 | 14.147 | 0.90 |
| exp008 | 13.81 | 12.339 | 1.47 |
| **exp026** | **10.06** | **8.672** | **1.39** |

- CV改善 3.75(13.81→10.06) が LB改善 3.67(12.34→8.67) に **ほぼ1:1転移**。gap安定~1.4。
- → **leak-freeローカルCVは採点テストに正しく一般化する**。以前の「test=3well/LBはノイズ」懸念は誤り。
  採点テストは大規模(数百well、kernel実行に~3h要した)でCVと整合する母数だった。

## 順位分析（2145チーム, board圧縮済み）
| ライン | score |
|---|---|
| 1位 | 6.534 |
| 1%tile | 8.121 |
| **我々** | **8.672 (上位2.9%)** |
| 10%tile | 8.863 |
| 中央 | 10.704 |

- 中位(exp008 54.6%)→**トップ3%**へ。PFトラッキングが決定打。
- 上位勢(6.5〜7.5)はリーク(physical model=train contacts)や強PF+多モデルアンサンブル併用と推定。
- 我々は純leak-free。健全な到達点。

## kernel実装メモ
- 自己完結: geom(LightGBM 5-fold) + PF(tuned: init_spread=4,pn=0.01, 128seed×500粒子) をrawから再計算→blend(0.683/0.392)+mean平滑(w101)。添付データ不使用。
- 高速化の教訓: numbaは np.random/np.interp呼び出しで逆に遅く棄却(10.8s/well)。ベクトル化単一スレッドも遅い(FLOP律速)。**マルチプロセスPF(ProcessPoolExecutor over wells)が正解**でCV完全同一(corr1.0)。
- exp020(echo kernel)はScoring Error: 3well分の事前予測を添付→採点再実行の大規模id集合と不一致。**自己完結が必須**の教訓。

## リンク
[[exp026_final_blend]] [[exp022_particle_filter]] [[submission_001_exp008]]
