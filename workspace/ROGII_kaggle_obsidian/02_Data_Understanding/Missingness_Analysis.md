# 欠損分析

既知の事実:

- `TVT_input` は各train wellで連続したtailとして欠損する。
- test submission rowsは `TVT_input` 欠損行と一致する。
- `GR` は欠損が多く、信号であると同時に不確実性として扱う。

必要な確認:

- wellごとの欠損率
- pre-PS / post-PS別の欠損率
- 欠損連続長
- GR欠損率とOOF errorの関係
