# Data Quality Checks

## Raw Inventory Checks

- train horizontal count と train typewell count が一致する。
- すべてのtrain wellにPNGがある。
- test horizontal count と test typewell count が一致する。
- sample submission IDs が test wells と row indices に対応する。

## Horizontal Checks

- required columnsが存在する。
- `MD` が単調増加している。
- `row_idx` が重複しない。
- `TVT_input` の欠損tailが連続している。
- trainではPrediction Start以前の `TVT_input` が `TVT` と一致する。
- target rowsが空ではない。
- GR missingnessを記録する。

## Typewell Checks

- required columnsが存在する。
- `TVT` と `GR` がnon-nullである。
- `TVT` range が関連するhorizontal wellのTVT rangeと重なる。
- duplicate TVT valuesを明示的に扱う。

## Train/Test Parity Checks

- model feature columns がtrain/test両方に存在する。
- test matrixにtarget columnを入れない。
- submission IDs が sample submission と完全一致する。

## 実験artifact checks

実験を信頼する前に確認すること:

- OOF row count が training target row count と一致する。
- fold column が期待foldを持つ。
- 同一fold内でvalidation wellがtrainingに入らない。
- submission row count が sample submission row count と一致する。
- submission IDs が sample submission と完全一致する。

## 現在のvalidation結果

2026-05-25時点:

- inventory: train horizontal 773、train typewell 773、test horizontal 3、test typewell 3。
- sample submission IDs: 14,151件でtest target IDsと完全一致。
- `TVT_input` missing tail violations: train 0、test 0。
- train known TVT check: 1,308,266 rows、max abs diff 0.0。
- fold file: 773 wells、5 folds。
- fold target rows: 757,738 / 756,650 / 756,255 / 757,101 / 756,245。
