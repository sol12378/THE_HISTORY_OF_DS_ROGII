# Alignment Features

## 仮説

horizontal GR window と typewell GR window を対応づけることで、Prediction Start以降のTVT driftを補正できる。

## 候補特徴量

- best GR correlation shift
- best matching TVT
- correlation score
- DTW distance
- lag between anchor baseline and GR-matched TVT
- local confidence based on GR coverage
