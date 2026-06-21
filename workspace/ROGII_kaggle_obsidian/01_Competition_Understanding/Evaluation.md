# 評価

公式metric:

```text
RMSE = sqrt(mean((manualTVT - predictedTVT)^2))
```

ローカルvalidationのルール:

- `TVT_input` が欠損している行だけを評価する。
- row単位ではなくwell単位でsplitする。
- すべての実験でOOF predictionを保存する。
- CV-LB gapとリークリスクを追跡する。
