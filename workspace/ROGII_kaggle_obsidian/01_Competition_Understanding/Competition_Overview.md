# コンペ概要

Prediction Start以降の水平井ポイントについて、欠損している `TVT` を予測する。

このタスクは以下を組み合わせる問題である。

- well path geometry
- Prediction Start以前の `TVT_input`
- horizontal wellの `GR`
- typewellの `TVT`-`GR` 参照曲線
- well単位の信頼できるvalidation

公式metricは提出した `tvt` に対するRMSE。
