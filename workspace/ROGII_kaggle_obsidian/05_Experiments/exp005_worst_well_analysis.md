# exp005_worst_well_analysis

## 目的

[[exp004_oof_error_slicing]] で特定したworst wells（727a3a10, 0390d174, b95e7121）の共通パターンを解明する。
`2000+` の long tail を細分化して、どの長さ帯でLightGBMが崩れるかを確認する。

## 仮説

- 「anchorが非常に強い（anchor_rmse < 10）かつdownward方向のwell」でLightGBMが過補正する。
- 8000+ の超長wellはanchorが最も安定しており、LightGBM deltaが邪魔になる。
- k-NN類似wellでも、同じ崩れパターンが潜在している可能性がある。

## 実装

- script: `scripts/exp005_worst_well_analysis.py`
- 入力: `experiments/exp003_lgb_anchor_trajectory/oof.csv`, `experiments/exp001_anchor_baseline/oof.csv`, `data/processed/train_base_v001.parquet`
- 出力: `experiments/exp005_worst_well_analysis/`

## Overall（参照元: exp003と同じ）

| metric | LightGBM | Anchor |
|---|---:|---:|
| RMSE | 15.054865 | 15.909853 |
| MAE | 10.655868 | 11.196480 |

## Long Tail 細分化

| bin | n_wells | model RMSE | anchor RMSE | improvement |
|---|---:|---:|---:|---:|
| 2000-2999 | 40 | 11.90 | 12.27 | +0.37 |
| 3000-3999 | 138 | 13.09 | 13.98 | +0.89 |
| 4000-4999 | 248 | 16.07 | 16.42 | +0.35 |
| 5000-5999 | 199 | 13.47 | 14.13 | +0.66 |
| 6000-7999 | 128 | 16.97 | 18.95 | +1.99 |
| **8000+** | **14** | **15.15** | **13.58** | **-1.57 ← LGB負け** |

解釈: 8000+だけがanchorに負けている。超長wellでは最後の既知TVTが非常に安定しており、LightGBMが足す微小deltaが誤差を増やす。4000-4999と5000-5999でimprovement小（0.35, 0.66）なのも警戒帯。

## anchor_blown wells (11件)

定義: anchor_rmse < 15 かつ LightGBMによるRMSE改善 < -10

| 特徴 | 値 |
|---|---|
| n_wells | 11 |
| mean hidden_length | 4703 |
| mean anchor_rmse | 9.0 |
| mean model_rmse | 31.0 |
| trajectory_shape | downward_curved 45.5%, upward_smooth 18.2%, flat_smooth 18.2% |

## focus well 3件の共通点

| well | hidden | anchor RMSE | model RMSE | diff | shape | delta_Z |
|---|---:|---:|---:|---:|---|---:|
| 727a3a10 | 4397 | 12.4 | 50.2 | -37.8 | flat_smooth | -160 |
| 0390d174 | 4411 | 8.5 | 42.4 | -33.9 | downward_curved | -178 |
| b95e7121 | 6306 | 3.4 | 32.9 | -29.5 | downward_curved | -165 |

共通点:
1. **anchor_rmse < 13** → anchorが非常に信頼できる
2. **final_delta_z < -160** → 強くdownward（実際にはほぼ垂直下降）
3. **hidden_length 4000超** → long tail帯
4. **known_length 1000-2000程度** → 短い既知区間、anchorが安定する条件

restとの最大の違いは `final_delta_z_from_ps`:
- focus wells 平均: -167 (強いdownward)
- rest 平均: +27 (むしろupward傾向)

## k-NN 類似well (上位3-4件)

**727a3a10の近傍:** flat_smooth、hidden 3500-5900。anchor_rmse 3-19。LightGBMはほぼ安全に動いているが一部(a8ed028a)はanchor_blown。

**0390d174の近傍:** downward_smooth/curved、hidden 4300-5400。anchor_rmse 6-16。LightGBMは概ね安全。

**b95e7121の近傍:** downward、hidden 4960-6920。anchor_rmse 5-14。rank 4の `8cc800b3` (anchor_blown=True, rmse_diff -11.7)が最も危険。

## 次アクション

1. **anchor_blown guard ルール設計**: `anchor_rmse_estimate < 10 かつ hidden_length > 4000` でLightGBM deltaを0に戻す（または0.3倍にshrink）するルールをexp006で試す。
2. **8000+ well 完全anchor化**: hidden_length > 8000 wellはanchor予測をそのまま使う後処理をexp006で試す。
3. **test set risk flag**: k-NNで特定したtest wells（downward, anchor_rmse推定 < 10, hidden 4000+）に事前フラグを立て、guard適用の準備をする。

## リーク懸念

exp003 OOF・exp001 anchor OOF・train base metadataのみ使用。学習には未使用。リーク懸念低。
ただし anchor_rmse をguard条件に使う場合、test setでは「提出前に anchor で予測してからguard判定」する実装が必要。
