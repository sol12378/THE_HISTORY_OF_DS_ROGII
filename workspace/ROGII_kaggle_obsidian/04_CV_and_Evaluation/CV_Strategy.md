# CV戦略

Primary validation:

```text
GroupKFold by well_id
```

Evaluation rows:

```text
TVT_input is NaN
```

理由:

- 同じwell内の隣接行は非常に相関が高い。
- row-random CVは局所的な連続性をリークする。
- test taskはfuture missing tailを予測するため、local CVもheld-out wellsを模倣すべきである。

Secondary checks:

- spatial holdout
- typewell similarity holdout
- GR missingness slices
- hidden-length slices
- trajectory-shape slices

## 推奨 v001 Fold

評価対象行数のバランスを取ったwell-level splitを使う。

```text
data/folds/folds_group_well_v001.csv
columns: split, well_id, fold
```

実装案:

- target rowsだけのrow-level recordsを作る。
- `GroupKFold(n_splits=5)` を `groups=well_id` で実行する。
- 得られたfold assignmentをwell単位に戻す。
- training rowsへ `well_id` でjoinする。

これにより、同じwellの全行を同じfoldに保ちながら、評価対象target rows数もfold間で揃えられる。

## Row Randomを使わない理由

row-random splitは無効である。理由:

- 隣接行はほぼ同じ `MD`, `X`, `Y`, `Z`, `GR` を持つ。
- 多くのwellでtarget `TVT` は滑らかに変化する。
- validation rowのすぐ隣がtrainに入る。
- scoreが未見wellへの汎化ではなく、局所補間の評価になる。

## Well数だけで単純KFoldしない理由

well数を均等にrandom splitする方法もbaselineとしては使えるが、wellごとのhidden lengthが異なるためtarget rows数が偏る。

ROGIIではwellごとのtarget rows数の幅が大きいため、評価行数をバランスさせる方が望ましい。

## 評価契約

すべてのCV reportに含めるもの:

- target rowsに対するfold RMSE
- foldごとのwell数
- foldごとのtarget rows数
- 全OOF target rowsに対するweighted overall RMSE
- well-level RMSEのunweighted mean
- bias: `mean(pred - TVT)`

## best_iteration の見方

LightGBMの `best_iteration` は、validation foldのRMSEが最も良かった木の本数を表す。

たとえば `n_estimators=600` でも、early stoppingで `best_iteration=124` なら「124本目までが最良で、その先はvalidationを改善しなかった」という意味になる。

解釈のポイント:

- 小さすぎる `best_iteration`
  - そのfoldでは特徴量が十分に効いていない
  - train / valid の分布差が大きい
  - learning_rateが大きすぎる可能性がある
- 大きすぎる `best_iteration`
  - まだ学習余地がある
  - `n_estimators` 上限に張り付いているなら上限不足の可能性がある

ROGIIでの実務ルール:

- foldごとに `best_iteration` を必ず保存する
- fold間で大きくぶれる場合は「CVの難しさの違い」か「特徴量の効き方の不安定さ」を疑う
- `best_iteration` 単独ではなく、fold RMSEと一緒に読む
- 採用判断は `best_iteration` のきれいさではなく、OOF sliceの一貫性で行う
