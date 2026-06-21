# exp043 Flagship Kernel

## 概要

ROGIIコンペ最終フラグシップkernelです。以下の要素を統合しています：

1. **exp026ベース**: PF + geom ブレンド、leak-free自己完結実装
2. **Multi-scale PF (P0)**: 温度 {3.0, 5.0, 8.0, 12.0} での尤度加重アンサンブル
3. **Residual GBDT (P1)**: PF残差学習（LGB/XGB/CB 5-fold GroupKFold）
4. **P2特徴**: 3D tortuosity + KNN formation surface距離
5. **最終ブレンド**: 0.76 * exp041_pf_residual_gbdt + 0.24 * geom

## パイプライン

```
train/test CSV読み込み
  ↓
ベース特徴量再構成 (Group A/B/C/D/F)
  ↓
P2特徴計算 (tortuosity + KNN)
  ↓
train: PF実行 → per-seed log-liks を多温度で再加重
  ↓
train: Residual GBDT学習 (fold-aware CV, 3-model ensemble)
  ↓
train: Geom LightGBM学習
  ↓
test: PF実行
  ↓
test: Residual GBDT予測
  ↓
test: Geom予測
  ↓
ブレンド: exp041(0.76) + geom(0.24)
  ↓
平滑化 (w=101, well内row順)
  ↓
submission.csv出力
```

## 推定実行時間（Kaggle上）

- PF train (773 wells, 128 seeds, 500 particles): ~36分
- Residual GBDT: ~15分
- Geom LightGBM: ~10分
- PF test (3 wells): ~2分
- GBDT/geom予測 + blend: ~5分
- **合計: ~70分 (1h10m想定)**

## Leak-free証明

- hidden TVT使用なし
- GR, typewell, anchor(last_known_TVT/X/Y/Z), Z, MD, X, Y のみ使用
- train既知点（TVT_input）の情報は学習に使用（当然）
- PF: 各well typewell GRプロファイル照合のみ
- GBDT: 基本特徴 + P2 + PF予測 + GR rolling（public CV対応）

## Self-contained実装

- `/kaggle/input/` からCSV読み込み
- 特徴量計算すべて このkernel内
- 事前学習モデル・外部データなし

## エラー処理

- PF失敗（typewell不足等）→ anchorで埋め替え
- GR欠損 → 補間 → グローバル平均で埋め替え
- XGBoost/CatBoost未install → LightGBMのみで代替

## ローカル検証結果

smoke test (minimal config): **✓ PASSED**
- 基本特徴量: ✓
- P2特徴: ✓
- PF (1 well): ✓
- Residual GBDT: ✓ (警告：計算コスト)
- ブレンド&平滑: ✓
- 提出形式: ✓ (14151行, NaN無し)

## 参照した既存実装

- `scripts/exp040_multiscale_pf.py`: Multi-scale PF温度アンサンブル
- `scripts/exp041_pf_residual_gbdt.py`: Residual GBDT + 3-model ensemble
- `scripts/features_p2.py`: P2特徴（tortuosity + KNN）
- `kaggle_notebooks/exp026_pf_geom_blend/rogii_exp026_pf_geom_blend.py`: Base特徴 + PF + geom

## Kaggle提出注意

- GPU不要
- インターネット接続不要
- データソース: rogii-wellbore-geology-prediction競争データのみ
- 実行時間上限: 9時間（余裕あり）
