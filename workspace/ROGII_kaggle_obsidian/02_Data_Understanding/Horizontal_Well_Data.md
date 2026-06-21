# Horizontal Well Data

Horizontal wellの各行は、井戸パス上の順序付きポイントを表す。

主要列:

- `MD`: well pathに沿った距離。
- `X`, `Y`, `Z`: 3次元位置。
- `GR`: gamma ray log。欠損が多い。
- `TVT_input`: Prediction Start以前は既知、以降は欠損。
- `TVT`: train target。

基本的なtraining target rows:

```text
TVT_input is NaN
```

Prediction Startは、`TVT_input` が初めて欠損する行である。
