# 用語集

## 中核用語

- `MD`: Measured Depth。井戸に沿って測った距離。
- `X`, `Y`, `Z`: 水平井の各点の3次元座標。
- `TVT`: True Vertical Thickness。今回の予測対象となる地質座標。
- `TVT_input`: Prediction Start以前に観測されているTVT。Prediction Start以降は欠損する。
- `GR`: Gamma Ray log。地層の指紋のように使えるログ値。
- `Typewell`: `TVT` と `GR` を持つ参照井。
- `Prediction Start`: `TVT_input` が初めて欠損する行。
- `OOF`: out-of-fold prediction。CVで得たvalidation予測。

## 地層ラベル

- `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`: train horizontal fileに含まれるformation marker名。
- `U` / `L`: geological labelにおけるupper / lower系の修飾と推定される。
- `TGT`: target interval系のラベル。
- `THL`, `BHL`: top/baseまたはlanding関連のinterval labelとして扱う。公式に確定できるまではラベル名として扱う。
