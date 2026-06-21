# exp004_oof_error_slicing

## 目的

[[exp003_lgb_anchor_trajectory]] のOOFを、well別・hidden_length別・trajectory形状別に分解し、CVを落としている原因を特定する。

## 仮説

LightGBMはoverallではanchorを改善するが、すべてのwellで安定して勝っているわけではない。特にlong tailや曲がりの強いtrajectoryでは、過補正によってanchorより悪化するwellがある。

## 実装

- script: `scripts/analyze_oof.py`
- target experiment: `exp003_lgb_anchor_trajectory`
- anchor baseline: `exp001_anchor_baseline`
- output: `experiments/exp004_oof_error_slicing/`

## Overall

| metric | LightGBM | Anchor | improvement |
|---|---:|---:|---:|
| RMSE | 15.054865 | 15.909853 | 0.854988 |
| MAE | 10.655868 | 11.196480 | 0.540611 |
| bias | -0.109339 | -1.595987 | - |
| p95 abs error | 30.689970 | 32.340000 | 1.650030 |

## Well別

- target wells: 773
- LightGBMがanchorより改善: 446 wells
- LightGBMがanchorより悪化: 327 wells
- RMSEがanchorより10以上悪化: 15 wells

Worst wells:

| well_id | fold | hidden_length | shape | model RMSE | anchor RMSE | diff |
|---|---:|---:|---|---:|---:|---:|
| 727a3a10 | 2 | 4397 | flat_smooth | 50.204930 | 12.363958 | -37.840972 |
| 0390d174 | 1 | 4411 | downward_curved | 42.380758 | 8.517417 | -33.863340 |
| b95e7121 | 3 | 6306 | downward_curved | 32.913595 | 3.420460 | -29.493135 |
| 65a5466a | 1 | 4683 | downward_curved | 37.752295 | 11.991020 | -25.761275 |
| a6f967fb | 3 | 4670 | flat_smooth | 38.943940 | 13.762814 | -25.181126 |

解釈:

全体改善の裏で、かなり明確な「過補正負け」が存在する。特にanchorが非常に強いwellに対して、LightGBMが大きなdeltaを足して崩しているケースがある。

## hidden_length別

| hidden_length_bin | rows | wells | model RMSE | anchor RMSE | improvement |
|---|---:|---:|---:|---:|---:|
| 200-499 | 407 | 1 | 1.765701 | 2.555310 | 0.789609 |
| 500-999 | 911 | 1 | 3.049902 | 3.601169 | 0.551268 |
| 1000-1999 | 7087 | 4 | 9.451801 | 15.319532 | 5.867732 |
| 2000+ | 3775584 | 767 | 15.065963 | 15.913596 | 0.847633 |

解釈:

train target rowsのほとんどが `2000+` に集中している。短いhidden_lengthの結果はwell数が少なく、一般化判断には使いにくい。現実的にはlong tail領域の中をさらに細かく切る必要がある。

## trajectory形状別

| shape | rows | wells | model RMSE | anchor RMSE | improvement |
|---|---:|---:|---:|---:|---:|
| downward_curved | 376762 | 69 | 16.316224 | 15.847869 | -0.468355 |
| downward_smooth | 747114 | 143 | 14.911755 | 15.824236 | 0.912480 |
| flat_curved | 345147 | 83 | 15.872798 | 16.580398 | 0.707600 |
| flat_smooth | 693701 | 172 | 15.336718 | 16.017655 | 0.680937 |
| upward_curved | 540230 | 103 | 14.583451 | 15.318506 | 0.735055 |
| upward_smooth | 1081035 | 203 | 14.464807 | 15.992225 | 1.527418 |

解釈:

`downward_curved` だけはshape単位でanchorより悪化している。曲がりの強いdownward trajectoryでは、現状のLightGBM deltaが安全ではない可能性がある。一方、`upward_smooth` は最も改善幅が大きく、この方向ではモデルがかなり効いている。

## 次アクション

1. `downward_curved` 向けにdeltaを弱める実験を行う。
2. worst 15 wellsを可視化し、anchorが強いケースの共通点を見る。
3. long tail `2000+` を `2000-3999`, `4000-5999`, `6000+` に再分割する。
4. anchorとLightGBMのblendを、shape別またはconfidence別に試す。

## リーク懸念

OOF予測とtrain base tableのtarget-row metadataだけを使った後処理分析。モデル学習には使っておらず、リーク懸念は低い。ここから特徴量やルールを作る場合は、testでも同じ定義で作れることを確認する。
