# リークしやすい特徴量

安易に使わないもの:

- `TVT`
- future `TVT_input`
- train-only formation marker columns
- train-only `Geology` labels
- leaked public test rowsを記憶するwell ID

許容される用途:

- 分析
- error slicing
- train targetの構築
- `leak_risk` と明記した管理下のリーク確認submission
