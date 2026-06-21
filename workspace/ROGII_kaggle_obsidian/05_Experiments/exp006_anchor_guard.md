# exp006_anchor_guard

## 目的

exp003 LightGBM の後処理として、ルールベースの anchor guard を実装し、
「anchor が強いのに LightGBM が大きく崩すwells」での破滅的な誤差を防ぐ。

## アプローチ（B案: ルールベース、test setで計算可能）

`pred = alpha * lgbm + (1 - alpha) * anchor` を well-levelで適用。

| rule | 条件 | alpha | n_wells |
|---|---|---|---|
| full_anchor | hidden_length > 8000 | 0.0 | 14 |
| delta_extreme | mean_pred_delta > 30 | 0.0 | 4 |
| strong_guard | final_delta_z < -100 AND hidden > 4000 | 0.7 | 228 |
| mild_guard | final_delta_z < -50 AND hidden > 4000 | 1.0 | 15 |
| no_guard | otherwise | 1.0 | 512 |

- `full_anchor`: 8000+帯はanchorが確実に勝つ（exp005で確認）
- `delta_extreme`: LGBMがper-well平均delta > 30を予測するwellは全て anchor比較でRMSE -25〜-38の壊滅的失敗
- `strong_guard` alpha=0.7: downward+long wellは僅かにanchor mix が効く
- `mild_guard`, `no_guard`: alpha=1.0（LGBMそのまま）

## 結果

| metric | 値 |
|---|---|
| LightGBM CV RMSE | 15.054865 |
| Blended CV RMSE | 14.724541 |
| improvement vs LightGBM | +0.330323 |
| improvement vs Anchor | +1.185311 |

## Per-rule 内訳

| rule | n_wells | blended RMSE | lgbm RMSE | anchor RMSE |
|---|---:|---:|---:|---:|
| full_anchor | 14 | 13.58 | 15.15 | 13.58 |
| delta_extreme | 4 | 9.37 | 40.52 | 9.37 |
| strong_guard | 228 | 14.58 | 14.60 | 15.70 |
| mild_guard | 15 | 24.14 | 24.14 | 26.37 |
| no_guard | 512 | 14.50 | 14.50 | 15.74 |

## 重要な発見

- **delta_extreme** が最大の改善源。4 wellsで lgbm RMSE 40.5 → blended 9.4 に激減。
- `mean_pred_delta > 30` は train全773wellsのうち4件のみ（極めてレア）。
- これらはすべて「anchorが非常に強い（anchor RMSE < 15）のにLGBMが大きな正のdeltaを予測する」ケース。
- 727a3a10（flat_smooth）も delta_extreme に含まれ、no_guard → delta_extreme でguard適用された。

## test 結果

3本のtest wellsはすべて no_guard（mean_pred_delta ≤ 30, hidden ≤ 8000）。
→ submission は exp003 と同一。**LB スコアは変わらない見込み。**
→ CV改善（+0.330）は train の信頼性改善として解釈する。

## 次アクション

1. **exp007**: trajectory diff特徴量（dX/dMD, dZ/dMD, pre_PS_TVT_slope）をLightGBMに追加し、内部で崩れを減らす。
2. **exp008**: GR rolling特徴量を追加（gr_rolling_mean, gr_diff_from_anchor）。
3. **exp009**: typewell GR alignment — 最大ポテンシャル（+2〜4 RMSE）。

## リーク懸念

- `hidden_length`, `delta_Z_from_PS`: train/testともに直接観測可能。リーク低。
- `mean_pred_delta`: モデルの予測値から計算。train/testで同じモデルを使う限りリーク低。
- alpha はOOFチューニング。候補7点の粗い探索のため過学習リスク低。
