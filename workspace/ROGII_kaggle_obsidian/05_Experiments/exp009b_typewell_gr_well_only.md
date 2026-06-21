# exp009b_typewell_gr_well_only

## 目的

exp009 の失敗原因が「per-row GR 補正のノイズ増幅」にあると仮定し、
per-row E 特徴量を全削除して per-well E 特徴量のみを残す。

仮説: exp009 分析より corr > 0.85 wells (243 wells) では exp009 > exp008 だったため、
ノイズを除去すれば全体改善が見込めると期待した。

## 変更点（exp009 との差分）

| 操作 | 特徴量 | 理由 |
|---|---|---|
| **削除** | gr_dev_from_tw_anchor | ノイズ入力 (0.6M gain) |
| **削除** | tw_tvt_correction | -gr_dev/slope でノイズ増幅 σ≈80 TVT (0.5M) |
| **削除** | tw_tvt_correction_reliable | 同上のマスク版 (0.8M) |
| **削除** | tw_gr_extrap_range | 不要な補助量 |
| **保持** | tw_gr_scale, tw_gr_offset | per-well GR較正 |
| **保持** | tw_known_gr_corr | typewell 信頼度指標 |
| **保持** | tw_gr_last20_diff | PS近傍の GR-typewell 乖離 |
| **保持** | tw_gr_slope_at_anchor | anchor点でのtypewell傾き |
| **保持** | tw_gr_abs_slope_at_anchor | 同絶対値 |
| **保持** | tw_gr_at_anchor | per-row 定数 (per-well 定数相当) |

## 結果

| metric | 値 |
|---|---|
| **exp009b CV RMSE** | **13.932154** |
| exp008 baseline | 13.808621 |
| exp009 (all E) | 13.922291 |
| vs exp008 | **-0.123534（悪化）** |
| vs exp009 | -0.009864（さらに悪化） |

## Fold 別 RMSE

| fold | rmse | best_iter |
|---|---:|---:|
| 0 | 13.139644 | 631 |
| 1 | 13.968884 | 269 |
| 2 | 13.971967 | 408 |
| 3 | 13.940404 | 735 |
| 4 | 14.602584 | 1284 |

**exp008 との比較**:
- fold 0: 13.040 → 13.140 （悪化 +0.100）
- fold 1: 14.042 → 13.969 （改善 -0.073）
- fold 2: 13.534 → 13.972 （悪化 +0.438）
- fold 3: 13.963 → 13.940 （微改善 -0.022）
- fold 4: 14.426 → 14.603 （悪化 +0.177）

fold 2 の大幅悪化 (+0.438) が全体 CV を押し下げている。

## 重要な発見：Group E per-well 特徴量は overfitting 原因

### feature importance は高い → しかし CV は悪化

| 特徴量 | importance_gain |
|---|---:|
| tw_gr_scale | 181M |
| tw_gr_offset | 161M |
| tw_gr_last20_diff | 160M |
| tw_gr_abs_slope_at_anchor | 124M |
| tw_known_gr_corr | 118M |
| tw_gr_slope_at_anchor | 108M |
| tw_gr_at_anchor | 106M |

importance は高いが CV は悪化。**importance 高い ≠ 汎化有効**の典型例。

### なぜ overfitting するか

1. **typewell-well 関係の分布シフト**: known 区間での GR 相関パターンが、
   hidden 区間でも同じとは限らない。地層が変わる = typewell との対応が崩れる。

2. **773 wells の高次元な identity 情報**: tw_gr_scale, tw_gr_offset は
   各 well に一意の連続値。LightGBM は訓練 well に過剰適応する split を見つけられる。

3. **GroupKFold の漏れ**: typewell_id は well_id と 1:1 対応のため、
   fold 内でのみ有効なパターンを覚えてしまう可能性。

### exp009 の「corr > 0.85 で改善」は誤解だった

exp009 分析時に「corr > 0.85 wells (243/773) では wt_delta = -0.215 (改善)」を観測。
しかし exp009b で per-row E を削除したら even worse になった。

解釈: exp009 での「per-row E ありで高 corr wells が改善」は、
モデルが corr 高の wells に対して per-row E の使い方を学習したため。
per-row E を抜くと per-well E の訓練信号自体が弱まった可能性。
あるいは per-well E 単体では訓練 well に overfitting するが、
per-row E の presence がある種の regularization になっていた可能性。

## 結論

**Group E（typewell GR alignment）は現在の実装では有害。**

- per-row E 削除でも改善しない → per-well E 自体が問題
- Group E 特徴量を全て封印し、exp008 (CV=13.808621) をベストとして維持

## 次アクション

Group E 以外の方向性に集中する:

1. **LB 提出（exp008）**: CV=13.808621 を LB で確認
2. **exp010: 別の特徴量グループ探索**
   - Group F: hidden_length や well 深さによる層別補正
   - Group G: 近傍 well の類似度（空間的クラスタリング）
   - Group H: 実際の地層モデルへの fitting（typewell の TVT alignment）
3. **exp010: n_estimators / regularization チューニング**
   - fold 4 が 1284 iter → まだ underfitting の可能性
   - num_leaves 増加、learning_rate 低下

## リーク防止確認

- キャリブレーション: is_known_tvt==True 行のみ ✅
- typewell は参照曲線（別物理井戸）、ラベルではない ✅
- GR は観測可能な物理計測値 ✅
- per-row 補正ロジックは完全削除 ✅

## 実験系列の進捗

```
exp003: 15.054865  (LGB baseline)
exp007: 13.867054  (+1.188 A+B+C trajectory)
exp007b: 13.853360 (+0.014 n_est=1500)
exp008: 13.808621  (+0.045 GR rolling) ← CURRENT BEST
exp009: 13.922291  (-0.114 typewell all E → 失敗)
exp009b: 13.932154 (-0.124 typewell per-well E only → さらに悪化 → Group E 封印)
```
