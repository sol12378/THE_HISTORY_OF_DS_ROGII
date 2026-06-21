# exp008_gr_rolling

## 目的

exp007b (CV=13.853360) に Group D（GR rolling 特徴量）を追加し、
GR の文脈情報が TVT 予測に貢献するか検証する。

## 追加特徴量（Group D）

### Group D_well（per-well、known区間のみから算出）

| 特徴量 | 定義 | リーク | importance_gain |
|---|---|---|---:|
| pre_ps_gr_available_frac | known行のGR有効割合 | ✅ 安全 | 191M ← **最重要GR特徴量** |
| pre_ps_gr_last20_mean | known末尾20行 GR 平均 | ✅ 安全 | 154M |
| pre_ps_gr_mean | known行 GR 平均 | ✅ 安全 | 151M |
| pre_ps_gr_trend | known行 GR 変化率 | ✅ 安全 | 147M |
| pre_ps_gr_std | known行 GR 標準偏差 | ✅ 安全 | 143M |

### Group D_row（per-row、因果ローリング）

| 特徴量 | 定義 | リーク | importance_gain |
|---|---|---|---:|
| gr_rolling_mean_w50 | 後ろ向き rolling mean (w=50) | ✅ 安全 | 13M |
| gr_rolling_mean_w20 | 後ろ向き rolling mean (w=20) | ✅ 安全 | 1M |
| gr_rolling_std_w20 | 後ろ向き rolling std (w=20) | ✅ 安全 | 0.3M |
| gr_z_score | (GR - mean) / std | ✅ 安全 | 0.1M |
| gr_vs_pre_ps_mean | GR - pre_ps_gr_mean | ✅ 安全 | 0.1M |

## 結果

| metric | 値 |
|---|---|
| **exp008 CV RMSE** | **13.808621** |
| exp007b (n=1500) | 13.853360 |
| vs exp007b | **+0.044739** |
| vs exp007 | **+0.058433** |

### Fold 別 RMSE

| fold | rmse | best_iter |
|---|---:|---:|
| 0 | 13.039683 | 286 |
| 1 | 14.041555 | 241 |
| 2 | 13.534087 | 255 |
| 3 | 13.962596 | 286 |
| 4 | 14.425618 | 839 |

fold 4 は 839 反復（1500 未満 → early stopping 有効）。

## 重要な発見

### `pre_ps_gr_available_frac` が最大の GR 寄与源

- GR の値そのものより「known区間でGRが何%取れているか」が重要。
- GRの計測率は掘削速度・地域・装置によって異なり、**ウェルのタイプを間接的に示す**可能性。
- これはリークではなく純粋な観測情報。

### per-row rolling GR は効果薄

- `gr_rolling_mean_w50`: 13M
- `gr_rolling_mean_w20`: 1M
- `gr_z_score` / `gr_vs_pre_ps_mean`: 0.1M（ほぼゼロ）

理由: GR の TVT 相関は well によって -0.41〜+0.07 と大きくばらつく。
**typewell alignment なし**では rolling GR はほぼ無意味。

## リーク防止（実装確認）

- `_compute_per_well_gr_features()`: `is_known_tvt==True` 行のみ使用 ✅
- `_add_perrow_gr_features()`: `sort_values(['well_id','row_idx'])` 後に rolling ✅
- rolling は `center=False`（Pandas デフォルト）→ 後ろ向きのみ ✅
- 欠損補完: `pre_ps_gr_mean` → `GLOBAL_GR_MEAN=87.75`（train のみから計算） ✅
- TVT / TVT_input は一切使用しない ✅

## 実験系列の進捗

```
exp003: 15.054865  (LGB baseline)
exp007: 13.867054  (+1.188 A+B+C trajectory features)
exp007b: 13.853360 (+0.014 n_estimators=1500)
exp008: 13.808621  (+0.045 GR rolling)
```

## 次アクション

1. **exp009: Typewell GR alignment**
   - GR per-row rolling がほぼ効かないことを確認 → typewell との GR パターンマッチングが必要
   - 推定 CV 改善: -2〜-4 RMSE（最大の改善源）
   - typewell_train_base_v001.parquet の GR カーブを使って well の地層位置を推定する

2. **LB 提出（exp008）**
   - 現 best CV = 13.808621。exp003 LB=14.147 からの改善を LB で確認。
