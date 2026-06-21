# exp011_typewell_leak_test

## 目的

exp009b の Group E 特徴量が importance 上位なのに CV を悪化させた理由として、
**typewell リーク**を疑い検証する。

exp009b ノートは「typewell_id は well_id と 1:1 対応」と推測していたが、これは**誤り**。
実データ確認の結果:

- **34 wells (773中) が同一 typewell 曲線を共有**（13 グループ、最大は 10 wells）
- うち **10 グループが複数 fold にまたがる** → well_id GroupKFold でリーク露出

Group E は typewell 由来なので、共有 typewell 経由で val 側が train 側の情報の
恩恵を受け、CV が楽観的になっている疑いがある。

## 設計（difference-in-differences）

リークを fold 構成ノイズから分離するため、**最小移動 fold (deduped fold)** を構築:

- 既存 well-fold を基準に、重複 typewell グループのメンバーを代表 fold へ強制集約（18 wells 移動）。
- 同一 deduped fold 上で 2 構成を学習:
  - `baseline` = exp008 特徴量（Group E なし）
  - `E` = exp009b 特徴量（Group E あり）
- 指標 = **gap の変化** `(E−baseline)_dedup − (E−baseline)_well`。
  fold 構成効果は両 arm に共通なので gap 差で相殺され、リーク分だけが残る。

## 結果

| fold 方式 | baseline (E無し) | E有り | E − baseline |
|---|---:|---:|---:|
| well-fold（既存） | 13.808621 | 13.932155 | **+0.123534** |
| deduped（typewell集約） | 13.651645 | 13.928077 | **+0.276432** |

- gap 変化 (deduped − well): **+0.152898**
- 判定: **LEAK_CONFIRMED**

### 読み方

- リークの無い baseline は deduped で **−0.157 改善**（移動 18 wells による fold 構成効果）。
- ところが E は deduped で **ほぼ不変 (−0.004)**。
- E が baseline と同じ −0.157 の恩恵を受けられなかった = **その分が共有 typewell リークだった**。
- 差分 (+0.153) がリークの大きさ。**well_id GroupKFold は E を約0.15 過大評価していた。**

### 独立した裏付け（exp010 診断）

- 共有 typewell wells (34): exp009b で **−0.182 改善**（楽観的に良い = リーク疑い）
- 非共有 wells (739): **+0.090 悪化**

2 つの独立な証拠（DiD と共有 well 診断）が同じ方向 → **リーク確定**。

## 結論

1. **Group E 完全封印を確定**。リーク除去後の評価では E は baseline を +0.276 悪化させる。
   well-fold で見えた +0.124 は leak で軽く見えていただけ。
2. **importance 高い ≠ 汎化に有効**（むしろ leak の温床）。
3. **well_id GroupKFold は共有 typewell でリークする**。
   今後 typewell 由来特徴量を一切使わないか、typewell-grouped fold を採用する。
   - 生成済み: `data/folds/folds_group_typewell_v001.csv`

## 注意・限界

- DiD は「fold 構成効果が両 arm で平行」を仮定。18 wells 移動で baseline が大きく動いた
  ため、厳密な平行性には不確実性が残る。ただし exp010 の共有 well 診断が独立に同方向を
  示すため、結論の頑健性は高い。
- deduped baseline 13.65 は fold 構成が偶然易しくなった値で、本物の改善ではない。

## リンク
- [[exp009b_typewell_gr_well_only]] / [[exp010_oof_slicing_fold24]]
- [[Decision_Log]] / [[Lessons_Learned]]
