# exp007b_guard

## メタ情報

| 項目 | 値 |
|---|---|
| Exp ID | exp007b_guard |
| 実施日 | 2026-05-28 |
| ベース実験 | [[exp007_traj_features]] |
| 参照実験 | [[exp006_anchor_guard]] |
| 出力先 | experiments/exp007b_guard/ |

## 目的と仮説

- exp007 fold4 が n_estimators=600 を使い切り underfitting → 1500 に増やす
- `|mean_pred_delta7| > 30` wells で catastrophic RMSE 38-48 の発見
- exp006 は正方向のみガード → 負方向も対称化する

## 変更内容

- n_estimators: 600 → **1500** (early_stopping=50 は同じ)
- ガードルール: `|mean_pred_delta| > 30 OR hidden_length > 8000` → alpha=0 (full anchor)

## 結果

| metric | value |
|---|---|
| LGB raw CV RMSE | **13.853360** |
| Blended CV RMSE | 13.985993 |
| exp007 CV RMSE | 13.867054 |
| raw 改善幅 vs exp007 | **+0.013694** |
| blend 変化幅 vs exp007 | -0.118939 (悪化) |

### fold別 (n_estimators=1500)

| fold | RMSE | best_iter |
|---|---|---|
| 0 | 13.061989 | 458 |
| 1 | 13.980590 | 270 |
| 2 | 13.618236 | 180 |
| 3 | 14.228974 | 514 |
| 4 | **14.339238** | **872** (→ 600 から改善) |

## 考察

### n_estimators=1500 の効果

fold4 の best_iteration が 600→872 に増加。RMSE も 14.405 → 14.339 に改善。
全体 raw CV も 13.867 → 13.853 へわずかに改善。

### guard が逆効果だった理由

`|mean_pred_delta| > 30` に該当する 17 wells で分析:

| | RMSE |
|---|---|
| LGB | 15.09 |
| anchor | 18.03 |
| blended (alpha=0) | 18.03 |

**LGB が anchor より明らかに優秀**なため、alpha=0 でガードすることで悪化した。

解釈: n_estimators=1500 で underfitting が改善され、exp007 で catastrophic だった予測が修正された可能性が高い。guard の前提条件（LGB が壊れている）がすでに成立していなかった。

### 結論

**本実験の best は LGB raw (13.853360)。guard は不採用。**

## リーク懸念

low。guard 条件の mean_pred_delta は OOF/test prediction のみ使用。

## 次の改善案

1. **n_estimators=1500 のみ submit**: LGB raw (CV=13.853360) で提出
2. **guard 閾値の見直し**: より高い閾値（|delta|>50 等）で実験するか guard 完全廃止
3. **fold4 高 RMSE wells の特徴分析**: fold4 の worst wells に特化した Group D 特徴量を検討
