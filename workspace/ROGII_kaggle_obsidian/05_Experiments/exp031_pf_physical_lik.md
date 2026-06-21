# exp031_pf_physical_lik — Multi-tw PF + Physical Likelihood

## 概要
地質資料§C-1/C-2に基づくPF尤度の物理化:
- well単位 GR z-score正規化 (KCl mud +20API、tool calibration差を加法的drift として除去)
- GR derivative (window=5) を尤度に加える (微分はdriftに不変)
- 尤度 = exp(-d_norm²/2) × exp(-0.5 W_DERIV d_deriv²/2)
- W_DERIV=0.5、K=3 candidates、S=32 seeds

## 結果
| 指標 | exp031 | exp022 |
|---|---:|---:|
| pooled CV | 20.379 | 11.024 |
| beat anchor | 375/773 | 587/773 |
| broken(>20) | 175 | 47 |

**単体は大幅悪化** だが:
- 47壊れwell中、新PFで復活(<=20): **23/47** (50%救出!)
- 誤差相関 vs exp022 = **0.333** (低い→相補的)
- 0.9 exp022 + 0.1 exp031 ブレンド = CV 10.78 (exp022単体11.02から−0.24)
- NNLS optimal (2-blend) = CV **10.396**

## 解釈
**正規化GRは raw GR と異なるwell集合を救う**:
- exp022 (raw GR) は drift小さい well で勝つ
- exp031 (normalized GR) は drift大きい well で勝つ
- 両者blendで相補効果

W_DERIV=0.5は強すぎた可能性。tune余地あり。

## ブレンド寄与
exp033 final blendで weight 0.057 (controlled、しかしfold5で全て>=0.035)。

## 次の改善案
- W_DERIV=0.1〜0.3でre-run (微分項を弱める)
- 「raw GR + normalized GR」を1 PFに統合 (2-channel likelihood)
- exp022の壊れ47wellに限定して physical-lik 適用 (gate)

## リンク
[[exp022_particle_filter]] [[exp030b_multi_tw_vec]] [[exp033_final_blend]]
