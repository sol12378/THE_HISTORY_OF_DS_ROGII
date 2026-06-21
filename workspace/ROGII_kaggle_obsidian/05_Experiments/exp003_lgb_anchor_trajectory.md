# exp003_lgb_anchor_trajectory

## 仮説

`last_known_TVT` からの差分 `TVT - last_known_TVT` は、well内の位置、軌跡、GR、PSからの距離である程度説明できる。したがって、単純anchorよりRMSEを下げられる。

## 実装

- script: `scripts/run_baseline_lgbm.py`
- model: `LightGBMRegressor`
- target: `TVT - last_known_TVT`
- prediction: `last_known_TVT + pred_delta`
- fold: well単位Group fold
- features: `MD`, `X`, `Y`, `Z`, `GR`, `is_gr_missing`, `n_rows_in_well`, `known_length`, `hidden_length`, `last_known_*`, `delta_*_from_PS`, `post_ps_step`, `row_frac`

## 結果

| metric | value |
|---|---:|
| overall CV RMSE | 15.054865 |
| fold mean RMSE | 15.042749 |
| fold std RMSE | 0.602631 |
| anchorからのRMSE改善 | 0.854988 |
| OOF rows | 3,783,989 |
| submission rows | 14,151 |
| Public LB | 14.147 |
| CV-LB gap | -0.908 |

fold別RMSE:

| fold | RMSE | best_iteration |
|---:|---:|---:|
| 0 | 14.884156 | 8 |
| 1 | 14.948613 | 124 |
| 2 | 14.044872 | 594 |
| 3 | 15.623081 | 593 |
| 4 | 15.713021 | 600 |

上位重要特徴量:

1. `known_length`
2. `last_known_Y`
3. `n_rows_in_well`
4. `delta_Z_from_PS`
5. `last_known_TVT`

## OOF比較

| metric | anchor | LightGBM |
|---|---:|---:|
| RMSE | 15.909853 | 15.054865 |
| MAE | 11.196480 | 10.655868 |
| p95 absolute error | 32.340000 | 30.689970 |
| bias | -1.595987 | -0.109339 |

- 行単位では 53.2% のtarget rowsでLightGBMがanchorより改善。
- well単位では 446 / 773 wells でLightGBMがanchorより改善。
- 一部wellでは大きく悪化しているため、次はwell別・hidden_length別・trajectory形状別の誤差分析が必要。

## 解釈

LightGBMは全体としてanchorの系統的な下振れbiasをかなり補正している。一方、fold 0 のbest_iterationが8で止まっており、foldごとの分布差がある。`known_length` や `n_rows_in_well` が強いことから、well長や観測区間の違いをモデルが大きく利用している。これは有効だが、private LBへの一般化ではfold設計とwell別error slicingで過信を避ける必要がある。

Public LBは14.147で、OOF CV 15.054865より約0.908良い。testは3 wellsに限られるため、public splitがtrain fold平均より易しい可能性がある。CVより良いこと自体は悪くないが、Public LBだけに合わせるとPrivateで崩れる危険がある。

## リーク懸念

目的変数 `TVT` は `TVT - last_known_TVT` の教師値とOOF評価にのみ使い、特徴量からは除外した。`is_target`、`is_known_tvt`、`fold`、`id`、`source_path` などのmarker/ID列も特徴量には入れていない。`last_known_*` と `delta_*_from_PS` はtarget開始前のanchorと各行の測定値に基づくため許容する。ただし、座標 `Z` がTVTに近い情報を持つ可能性があるため、次に `Z` 系特徴量のablationを行う。

## 次アクション

1. `Z`, `delta_Z_from_PS`, `last_known_Z` を抜いたablation。
2. well別に悪化したケースを抽出し、trajectoryとGRの形状を見る。
3. hidden_length / known_length / fold別のerror slicingを定常レポート化する。
