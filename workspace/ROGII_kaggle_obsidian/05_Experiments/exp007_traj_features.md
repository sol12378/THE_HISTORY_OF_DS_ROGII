# exp007_traj_features

## 目的

exp003 LightGBM baseline (CV=15.054865) に Groups A+B+C 特徴量を追加し、
軌跡方向・地層傾斜・ウェル形状の情報を LightGBM に与えることで精度改善を図る。

## 追加特徴量

### Group A: Pre-PS TVT momentum（地層傾斜モメンタム）

既知区間の TVT_input 末尾から計算する per-well 定数。

| 特徴量 | 定義 |
|---|---|
| pre_ps_tvt_slope_last20 | (TVT[-1] - TVT[-20]) / (MD[-1] - MD[-20]) |
| pre_ps_tvt_slope_last5 | (TVT[-1] - TVT[-5]) / (MD[-1] - MD[-5]) |
| pre_ps_tvt_curvature | slope_last5 - slope_last20（傾きの変化率） |
| pre_ps_tvt_delta_last20 | TVT[-1] - TVT[-20]（絶対変化量） |

### Group B: Trajectory direction（軌跡方向）

**Pre-PS per-well**（既知区間末尾20行の平均方向）:

| 特徴量 | 定義 |
|---|---|
| pre_ps_dZ_dMD | (Z[-1] - Z[-20]) / MD_diff |
| pre_ps_dX_dMD | (X[-1] - X[-20]) / MD_diff |
| pre_ps_dY_dMD | (Y[-1] - Y[-20]) / MD_diff |
| pre_ps_horiz_dMD | sqrt(dX²+dY²) / MD_diff |
| pre_ps_azimuth | atan2(dY, dX) |

**Per-row from PS**（各行のPS起点方向）:

| 特徴量 | 定義 |
|---|---|
| dZ_dMD_from_ps | delta_Z_from_PS / delta_MD_from_PS |
| dX_dMD_from_ps | delta_X_from_PS / delta_MD_from_PS |
| dY_dMD_from_ps | delta_Y_from_PS / delta_MD_from_PS |
| horiz_disp_from_ps | sqrt(delta_X²+delta_Y²) |
| azimuth_from_ps | atan2(delta_Y_from_PS, delta_X_from_PS) |

### Group C: Well shape ratios（形状比）

| 特徴量 | 定義 |
|---|---|
| kh_ratio | known_length / hidden_length |
| hidden_frac | hidden_length / n_rows_in_well |

## 結果

| metric | 値 |
|---|---|
| LightGBM CV RMSE (exp007) | 13.867054 |
| exp003 baseline | 15.054865 |
| 改善幅 | **+1.187811** |
| fold mean | 13.858821 |
| fold std | 0.474399 |

### Fold 別 RMSE

| fold | rmse | best_iteration |
|---|---:|---:|
| 0 | 13.061989 | 458 |
| 1 | 13.980590 | 270 |
| 2 | 13.618236 | 180 |
| 3 | 14.228974 | 514 |
| 4 | 14.405314 | 600 |

fold 4 が early stopping に達しなかった（n_estimators=600 使い切り）。
モデルが若干 underfitting の可能性。n_estimators 増加で更なる改善余地あり。

## Feature Importance Top 10 (gain)

| feature | importance_gain |
|---|---:|
| dZ_dMD_from_ps | 797M ← **新特徴量 #1** |
| last_known_Y | 367M |
| known_length | 354M |
| pre_ps_dZ_dMD | 344M ← **新特徴量** |
| pre_ps_tvt_curvature | 338M ← **新特徴量** |
| Y | 304M |
| X | 299M |
| pre_ps_tvt_slope_last20 | 282M ← **新特徴量** |
| delta_X_from_PS | 257M |
| delta_Z_from_PS | 253M |

## 重要な発見

- **dZ_dMD_from_ps が断然 #1**。PSからの方向が TVT の伸び方を支配している。
  - delta_Z_from_PS は以前からあったが、MD で正規化することで gain が約 1.7 倍に跳ね上がった。
- **pre_ps_dZ_dMD**（既知区間の鉛直方向）も #4。PS直前の軌跡方向は予測に直結。
- **pre_ps_tvt_curvature**（傾きの変化率）が #5。地層が加速/減速して変化しているかどうかが重要。
- GR は依然として最下位グループ（GR gain = 5.5M）。typewell alignment なしではほぼ寄与しない。

## リーク懸念

- Group A, B pre-PS は is_known_tvt 行のデータのみ使用。PSより前の観測値のみ → リーク低。
- Group B per-row は delta_*_from_PS（PSからの相対位置）から計算 → 既存特徴量と同等リスク（低〜中）。
- Group C は既存 known_length/hidden_length から計算 → リーク低。

## OOF 詳細分析

### well 単位の改善/悪化

| 区分 | well 数 | 平均Δ RMSE |
|---|---:|---:|
| 改善 (exp007 < exp003) | 473 | -3.773 |
| 悪化 (exp007 > exp003) | 300 | +3.008 |
| 悪化 > 5 RMSE | 53 | - |
| 悪化 > 10 RMSE | 9 | - |
| 悪化 > 20 RMSE | 3 | - |

### 「方向逆」問題

- exp007 が TVT 変化の**方向を逆**に予測するwellが **281件（36.4%）** 存在。
- 「方向一致 (492 wells)」: avg -2.029 RMSE 改善
- 「方向逆 (281 wells)」: avg +0.413 RMSE 悪化（軽微）
- ただし7件の極端な悪化ケース (delta > 10) は全て方向逆に起因。

### 方向逆の原因

pre_ps_tvt_slope（PSより前の地層傾向）を LightGBM が外挿するが、
**PSを境に地層傾向が逆転するwell**では外挿が大きく外れる。

具体例（27c3155b）:
```
pre_ps_tvt_slope_last20 = +0.027  (PSより前はTVT増加中)
→ exp007: TVT = 12758 〜 12797  (last_known_TVT=12756 から増加方向に予測)
→ 真値:   TVT = 12733 〜 12757  (実際はTVT減少)
→ RMSE exp003: 7.4  →  exp007: 48.5
```

### Anchor guard 適用による仮想改善

anchor より 20以上 悪い 5 wells を anchor に置き換えた場合:
- 仮想CV: **13.543**（現13.867 から -0.324）

### Fold 4 underfitting

fold 4 の best_iteration = 600（上限に到達）。他 fold は 180〜514。
→ n_estimators を増やすことで fold 4 の改善余地あり。

## 次アクション

1. **exp007b**: anchor guard 拡張 + n_estimators=1500
   - `|mean_pred_delta7| > 30` → anchor に置換（現在は正方向のみ）
   - fold 4 の underfitting を n_estimators 増で改善
   - 仮想 CV 目標: ~13.5

2. **LB 提出**: exp007 submission を提出してLBスコアを確認
   - exp003 (LB=14.147) との差分で trajectory 特徴量の汎化力を検証

3. **exp008**: GR rolling 特徴量（gr_rolling_mean_w50, gr_diff_from_anchor）
   - GR は依然として最下位 (importance gain 5.5M)。typewell alignment なしでは限界。

4. **exp009**: Typewell GR alignment — 最大ポテンシャル（推定 CV -2〜-4）
