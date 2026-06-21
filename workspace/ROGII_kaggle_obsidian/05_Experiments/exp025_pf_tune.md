# exp025_pf_tune — Particle Filter パラメータ調整

## 概要
exp022 PF(参照デフォルト)を起点に init_spread / PN(運動雑音) / scale(seed尤度温度) を grid 探索。
計算量のため **stratified ~129 well subset(hidden_length四分位)+ n_seeds=64** で探索し、勝者を full773 再実行(exp025b)→再ブレンド(exp026)。
params は payload に格納し ProcessPoolExecutor 並列(worker再import問題を回避)。完全leak-free。

## subset grid 結果（n_seeds=64, 129 well）
| config | subset CV |
|---|---|
| baseline (ispr2, pn0.005, scale8) | 10.726 |
| **ispr4_pn0.01** | **10.468** ← best (−0.258) |
| pn0.02 | 10.560 |
| np800 (粒子800) | 10.564 |
| scale5 | 10.762 |
| ispr4 | 10.769 |
| ispr1 | 10.728 |
| scale12 | 10.783 |
| pn0.01 | 10.796 |

## 勝因
- **init_spread=4.0**(初期粒子の広がり) + **PN=0.01**(行毎の運動雑音up)の組合せが最良。
  探索性が増し、PS直後のTVTジャンプや地層変化への追従が改善。単独(ispr4のみ/pn0.01のみ)は効かず、
  **組合せで初めて効く**(相互作用)。
- scale(seed尤度加重温度)は5/8/12で差小→8で十分。

## 次
- exp025b: ispr4_pn0.01 を full773・n_seeds=128 で再実行(tuned PF)。
- exp026: tuned PF × geom × trees/nn/attn を再ブレンド→最終CV。

## 注意（実装の落とし穴）
ProcessPoolExecutorに別モジュール(importlib読込)のworker関数を渡すと、macOS spawnで
子プロセスがimportできずデッドロックする。exp025b/025は worker関数を実行スクリプト自身に定義して回避。

## リンク
[[exp022_particle_filter]] [[exp024_multistage_blend]] [[exp026_final_blend]]
