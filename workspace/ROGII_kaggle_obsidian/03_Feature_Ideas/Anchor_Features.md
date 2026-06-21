# Anchor Features

## 仮説

Prediction Start直前の `TVT_input` は、未来のTVTを予測するうえで最も強い情報である。

## 候補特徴量

- `last_known_TVT`
- `last_known_MD`
- `last_known_X`
- `last_known_Y`
- `last_known_Z`
- `delta_MD_from_PS`
- `delta_X_from_PS`
- `delta_Y_from_PS`
- `delta_Z_from_PS`
- `pre_PS_TVT_slope`
- `pre_PS_TVT_curvature`

## Target Transform

推奨:

```text
target_delta = TVT - last_known_TVT
```
