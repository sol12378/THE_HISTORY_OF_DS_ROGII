# ROGII Next Autonomous Experiment Plan

作成日: 2026-06-21

## 現在地

自律研究基盤の初期パイプラインは完走済みである。

- `exp100_data_contract`: raw/sample/test-tail contractを確認し、processed base tableを作成済み。
- `exp110_group_well_cv_v2`: 773 wellのbalanced group-well 5-foldを作成済み。
- `exp120_anchor_repro`: last-known TVT anchor baselineを再現済み。CV RMSEは `15.909852870734554`。
- `exp120_anchor_repro_oof_slices`: fold別、well別の誤差診断を作成済み。

この段階で「自律実行の健全性」は確認できた。次は、スコア改善を狙うモデル系列を漏洩監査つきで再現し、OOF診断から次の仮説を選ぶ。

## 基本方針

ROGIIはwell trajectory、GR log、typewellとの対応、地層形状の外挿が絡む時系列・地質推定タスクである。今後の実験は次の順で進める。

1. Anchorを基準線として固定する。
2. GR rolling特徴量を使うtree modelで、ラベルを直接使わない局所シグナルの寄与を測る。
3. Typewell GR alignment / particle filter系列で、GR sequenceと層準の対応を陽に使う。
4. Geometry / tree / NN / PFをOOFでblendし、foldごとの過学習を検査する。
5. Worst wellsを継続的に抽出し、改善が全体平均だけでなく難井戸にも効いているかを見る。

## 今回起動する自律実験

### Phase 1: GR rolling tree baseline

Action: `exp008_gr_rolling`

目的:

- pre-PS既知区間からのTVT形状特徴量、well geometry特徴量、GR rolling特徴量をLightGBMに投入する。
- Anchorとの差分を学習対象にし、絶対TVTを直接追わないことで外挿を安定させる。
- `exp120_anchor_repro` からの改善幅とfold別の崩れを確認する。

判定:

- `result.json` が作成される。
- `oof.csv` と `submission.csv` が作成される。
- CV RMSEがanchorの `15.909852870734554` を下回るか確認する。
- `oof_slices` でworst wellsを更新する。

### Phase 2: Particle filter reproduction

Action: `exp022_particle_filter`

目的:

- GR sequenceとtypewell GRを用いた粒子フィルタ系の再現性を確認する。
- Tree modelでは拾いづらい連続的な層準追跡の寄与を見る。

判定:

- `per_well.csv`、`result.json`、可能ならOOF/submissionが作成される。
- 実行時間が長いため、ログとハートビートで停止・失敗を検出する。

### Phase 3: Final blend reproduction

Action: `exp026_final_blend`

目的:

- PF、geometry、tree、NN、attention系の既存OOFをNNLS nested-foldでblendする。
- 単体モデルよりfold安定性が上がるかを見る。

判定:

- `nested_plus_smooth_cv` を主指標にする。
- submission形式チェックを通す。
- blend weightが極端に1モデルへ寄っていないか確認する。

## Go / No-Go

Go:

- 実験が正常終了し、OOFとsubmissionが生成される。
- CVがanchorより改善、または特定fold/wellで有意な改善がある。
- leakage riskが低い、またはリスクが明示的に隔離されている。

No-Go:

- public overlapに依存した性能改善しか説明できない。
- fold別に1 foldだけ極端に改善し、他foldで悪化する。
- submission対象行数 `14151` を満たさない。
- OOF診断が生成できない。

## 次の判断

今回のバックグラウンド実行後、最新の `result.json` と `oof_slices` を読み、次のいずれかに進む。

- GR rollingが改善: 特徴量群のablation、fold v002への移植、worst wells向けpostprocess。
- PFが改善: particle数、seed数、GR likelihood、geometry priorの軽量チューニング。
- Blendが改善: leak-free nested blendの再実行、submission gate、候補提出ファイルの固定。
- どれも改善しない: exp120 anchor診断に戻り、worst wells別に地質・軌跡特徴の失敗原因を分類する。
