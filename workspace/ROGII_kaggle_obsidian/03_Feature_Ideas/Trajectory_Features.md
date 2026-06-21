# Trajectory Features

## 仮説

井戸の進行方向や曲率は、Prediction Start以降にTVTが増えるか、減るか、ほぼ一定かを説明する。

## 候補特徴量

- `dX_dMD`
- `dY_dMD`
- `dZ_dMD`
- azimuth
- inclination proxy
- horizontal displacement from PS
- vertical displacement from PS
- curvature
- dogleg-like local change
