# exp017 — GR-typewell照合 特徴量化（最終GR実験）

## 一言
GR-typewell照合を「学習特徴量(Group H)」として最大限投入し、leak-safe typewell-fold で評価。
**-0.067 悪化**。oracle診断と合わせ **CV<=5 は GR含む正規手段で到達不可能** を確定。

## 数値
| feature set (typewell-grouped fold) | CV |
|---|---:|
| base = exp014 全特徴 | 13.762346 |
| base + Group H (GR alignment) | 13.829807 |
| Group H 寄与 | **-0.067461** |

### oracle 上限（diag_gr_ceiling）
- glob_oracle (1 offset/well) = 8.21 / seg_oracle (4 offset/well) = 4.39
- corr(GR選択offset, 真offset) = +0.155（極弱）

## 解釈
- geom(exp014)の TVT(MD) **形状は正しい**。誤差は低周波 offset。理想offsetを当てれば CV<=5 も構造上可能。
- だが offset を当てる信号が GR に乏しい（corr 0.155）。GRを特徴量化しても leak-safe fold で過学習し悪化。
- exp009/011/013/013b と完全に整合。GR照合は全形態で NO-GO 確定。

## 成果物
- code: `scripts/exp017_gr_align_features.py`, `scripts/diag_*.py`
- results: `experiments/exp017_gr_align_features/`, `experiments/diag_gr_ceiling/`

## リンク
[[gr-offset-ceiling]] [[exp014_geom_extrap]] [[exp011_typewell_leak_test]] [[Strategy_2026-05-31]] [[Decision_Log]]
