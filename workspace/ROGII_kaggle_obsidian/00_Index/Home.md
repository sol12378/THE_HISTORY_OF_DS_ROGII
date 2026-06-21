# ROGII Kaggle Home

## 現在の状態

- Competition: ROGII - Wellbore Geology Prediction
- Code repository: project root
- Raw data: `data/raw/`
- 現在の優先事項: 安定したデータエンジニアリング基盤とCV基盤を作る。

## ナビゲーション

- [[Competition_Map]]
- [[Glossary]]
- [[Data_Structure]]
- [[CV_Strategy]]
- [[Data_Engineering_Overview]]
- [[Long_Term_Data_Engineering_Plan]]
- [[EXP_Index]]
- [[Decision_Log]]
- [[LB_Tracking]]

## 現在のベスト

| 種別 | Exp | CV | LB | Notes |
|---|---|---:|---:|---|
| CV | - | - | - | まだ未確立 |
| LB | - | - | - | leak lookup は隔離して扱う |

## 次のアクション

- base tableを作る。
- well単位のGroupKFoldを作る。
- anchor baseline と slope baseline を実行する。
- OOFベースの誤差分析を始める。
