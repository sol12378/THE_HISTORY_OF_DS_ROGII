# exp022_particle_filter — Particle Filter GR-typewellトラッカー（ブレイクスルー）

## 概要
参照notebook(ajayrao43/biohack44)の Particle Filter を忠実移植し、**全773 well の正直なleak-free CV**を算出。
状態 pos=TVT+Z(Z既知分を分離)をMD沿いに滑らか伝播、観測GR vs typewell GR(pos-Z) の尤度で粒子再重み付け。
128 seed × 500 粒子, 尤度加重(scale=8)。GRはwell全体を補間。well単位ProcessPoolExecutor並列(37分)。

## 結果（773-well pooled CV）
| 手法 | CV | 3 test well |
|---|---|---|
| anchor | 15.910 | 7.45/15.26/7.92 |
| Beam(exp021) | 15.697 | 8.38/14.78/8.49 |
| **PF** | **11.024** | **3.99/4.44/6.52(平均≈5.0)** |
| geom(exp014) | 13.525 | |
| blend(exp020) | 13.321 | |

PF が anchor に勝つ: 587/773。per-well RMSE 中央値5.66。47 well壊れ(>20)。

## 意義（並行作業の結論を覆す）
- **PF pooled CV 11.02 が全手法の最良**。完全leak-free（GR+typewell+anchor+Z+MDのみ、hidden TVT不使用）。
- 3 test well平均≈5.0 = 参照の"local 4.71 ft"を再現。**参照LB 6.8 はリークでなくPFトラッキング由来**
  （leak=0だがLB#1=6.8≠0 → train TVT≠採点真値、PFが本筋）。
- 並行作業(exp017 GR特徴量/exp019 系列NN/exp020 微分可能DTW)の「GR照合 全形態NO-GO・到達不可」は
  **学習的/ソフト/グローバル**形態のみ。PFは**well単位確率的ハードトラッカー**＝未検証の形態で、各wellが
  自分のtypewellを直接利用するため根本的に強い。→ NO-GOは部分撤回。
- なぜPF>Beam: 連続状態+滑らかrate運動+128seed尤度加重で多価GRの曖昧性を確率的統合。
  決定論ビームの±2制約より柔軟で迷子復帰可。

## ブレンド（次の本線）
- PF×geom 誤差相関0.426と低い → **0.6·PF+0.4·geom+平滑 = CV 10.16**(前best13.32から+3.16、全fold一貫9.49〜11.07)。
- 残課題: 47壊れwellのreliabilityゲート(PF log-lik)、tree/NN含む多段blend、PFパラメータ調整、Kaggle kernelでLB転移検証。

## リンク
[[exp021_beam_track]] [[exp023_leak_lookup]] [[exp014_geom_extrap]] [[Decision_Log]] [[gr-offset-ceiling]]
