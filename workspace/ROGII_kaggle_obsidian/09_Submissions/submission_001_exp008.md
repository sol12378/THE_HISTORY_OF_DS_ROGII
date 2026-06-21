# Submission 001 — exp008_gr_rolling

## 提出情報

| 項目 | 値 |
|---|---|
| 日時 | 2026-05-30 |
| exp | exp008_gr_rolling |
| model | LightGBM delta regression |
| features | SAFE + Group A/B/C/D |
| CV RMSE | 13.808621 |
| **Public LB** | **12.339** |
| rank | 1041 / 1906 (54.6%) |
| Kaggle kernel | sol12378/rogii-exp008-gr-rolling |

## CV-LB gap 分析

| exp | CV | LB | gap (CV−LB) | LBがCVより |
|---|---:|---:|---:|---|
| exp003 | 15.054865 | 14.147 | 0.908 | 0.908 良い |
| **exp008** | **13.808621** | **12.339** | **1.469** | **1.469 良い** |

- gap が exp003 (0.908) → exp008 (1.469) に拡大。Group A/B/C/D の特徴量が train CV より test wells に対してより有効に働いている。
- **CVの改善がLBに確実に転化する**ことが実証された。

## LB分布上の位置

| ライン | rank | score |
|---|---:|---:|
| 1位 | 1 | 6.828 |
| top10% | 190 | 9.491 |
| top25% | 476 | 9.745 |
| top50% | 953 | 10.974 |
| **我々** | **1041** | **12.339** |
| 旧 (exp003) | 1263 | 14.147 |
| 最下位 | 1906 | — |

top10%まで残り: 12.339 − 9.491 = **2.848**（大きな改善余地あり）

## 含意と方針

1. **CV-drivenで攻める**: CV改善1.246 → LB改善1.808。LBの方が改善幅が大きい。
   rough換算: LB ≈ CV − 1.5 (exp008時点)
2. **次のCV目標**: 13.808 → 13.0台 → 12.x台 → ゆくゆく top10% (≈8.x)
3. **特徴量の方向**:
   - Group F: post-PS trajectory外挿 (typewell不要、物理ベース)
   - anchor guard × exp008 (長尺hard well対策)
   - モデル多様化 (XGB/LGBM blend)

## 注意事項

- Public LBはtest wellsの一部（public fraction）。Private = 最終評価。
- CV-LB gap は今後特徴量が変わると変動する可能性。毎回測るのではなく定点でのみLB測定。
- exp003 とのgap差が大きいのは Group A/B/C/D の generalization が良いためと考えられるが、
  public test wellsの特性バイアスも混入している可能性あり（過信しない）。

## リンク
- [[exp008_gr_rolling]] / [[EXP_Index]]
