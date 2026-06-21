# データ構造

各wellにはペアのファイルがある。

```text
{well_id}__horizontal_well.csv
{well_id}__typewell.csv
```

trainにはさらに画像がある。

```text
{well_id}.png
```

Horizontal train columns:

```text
MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, TVT, GR, TVT_input
```

Horizontal test columns:

```text
MD, X, Y, Z, GR, TVT_input
```

Typewell train columns:

```text
TVT, GR, Geology
```

Typewell test columns:

```text
TVT, GR
```
