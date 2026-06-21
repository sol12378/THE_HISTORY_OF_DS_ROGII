# OOF分析ガイド

すべての実験でOOFに以下を保存する。

- `well_id`
- `row_idx`
- `fold`
- `TVT`
- `pred`
- `error`
- `abs_error`
- `TVT_input_isna`
- `last_known_TVT`
- `pred_delta`
- `known_length`
- `hidden_length`
- `post_ps_step`
- `row_frac`
- `delta_MD_from_PS`
- `delta_X_from_PS`
- `delta_Y_from_PS`
- `delta_Z_from_PS`

必須集計:

- overall RMSE
- fold RMSE
- well RMSE
- bias
- Prediction Startからの距離別error
- GR欠損率別error

## 目的

OOF分析の目的は「CVが良かった」で終わらせず、どの種類のwellで勝ち、どの種類のwellで負けたかを特定することにある。

## 分析の優先順位

1. well別誤差
2. hidden_length別誤差
3. trajectory形状別誤差

この順に見る。理由は、ROGIIでは最終的な失点が特定wellに集中しやすく、その背景としてhidden tailの長さや軌跡形状が効いている可能性が高いから。

## 1. well別誤差

必ず出す集計:

- wellごとのRMSE
- wellごとのMAE
- wellごとのbias
- wellごとのtarget rows数
- anchor比改善量

見る観点:

- 一部のwellだけ極端に悪いか
- anchorには勝つが分散が大きいwellがあるか
- foldごとに難しいwellが偏っていないか

実務ルール:

- overall CVだけでなく `well-level RMSE mean` も追う
- worst 20 wells を毎回固定で保存する
- 改善well / 悪化well をanchor比較で分ける

## 2. hidden_length別誤差

hidden_lengthは「どれだけ先を予測するか」の難しさそのものなので、必須のsliceとする。

推奨bin:

- `1-199`
- `200-499`
- `500-999`
- `1000-1999`
- `2000+`

必ず出す集計:

- binごとのRMSE
- binごとのMAE
- binごとのbias
- binごとのwell数
- binごとのtarget rows数

見る観点:

- tailが長いほど単純に崩れるか
- CV改善が短いtailだけに偏っていないか
- long tail well で anchor の方が安全でないか

実務ルール:

- 新特徴量は hidden_length short / medium / long の3領域で効き方を比較する
- overall改善よりも long tail の悪化を重く見る

## 3. trajectory形状別誤差

trajectory形状は「TVTの変化をどれだけ幾何的に説明できるか」をみるための軸である。

まずは複雑なクラスタリングより、解釈しやすい手設計分類から始める。

第一段階で作るshape指標:

- `mean_abs_dz_step`
- `std_dz_step`
- `mean_abs_dxdy_step`
- `final_delta_z_from_ps`
- `curvature_proxy`
- `azimuth_change_proxy`

基本分類案:

- `flat` : `|final_delta_z_from_ps|` が小さい
- `upward` : `final_delta_z_from_ps` が正に大きい
- `downward` : `final_delta_z_from_ps` が負に大きい
- `smooth` : `curvature_proxy` が小さい
- `curved` : `curvature_proxy` が大きい

必ず出す集計:

- shape classごとのRMSE
- shape classごとのbias
- shape classごとのwell数
- anchor比改善量

見る観点:

- 曲がりが強いwellでtree modelが本当に効いているか
- `Z` 系特徴量が特定shapeだけで効いていないか
- Public LBで良くても特殊shapeを落としていないか

## 比較ルール

比較対象は最低でも以下の2本に固定する。

- anchor baseline
- 現在のbest CV model

新実験では「anchor比」と「best model比」の両方を出す。これで、ただ複雑化しただけなのか、本当に弱点を潰したのかが分かる。

## 意思決定ルール

- overall RMSEだけ改善: 採用保留
- overall改善かつ worst wells が改善: 採用候補
- overall改善でも long tail / curved wells が崩壊: 原則不採用
- Public LBだけ改善でCV sliceが悪化: 不採用寄り
