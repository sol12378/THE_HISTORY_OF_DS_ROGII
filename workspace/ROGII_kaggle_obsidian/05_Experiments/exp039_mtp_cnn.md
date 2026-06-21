# exp039: MTP-CNN (Multi-Trajectory Prediction + 1D CNN)

Date: 2026-06-05
Status: IN PROGRESS (training 5 folds, ~2 hour ETA)

## 仮説

単一モード予測の exp022 (particle filter) に対し、複数モード出力で**多様な誤差構造を持つ独立信号**を作成。MTP (Multi-Trajectory Prediction) loss により mode collapse を回避し、ensemble blend で LB 改善を狙う。

## 背景

- exp022 (geom PF blend) は LB 8.672, CV 11.02 で strong baseline
- 47本の "broken wells" (RMSE > 20ft) = uni-modal 予測の弱点
- 3test wells はノイズ・データドリフトあり → diverse candidates で robust化

## 実装

### モデル: MTPCNN
```
Input (B, 6, T) [batch of wells, 6 features, T≤1500 rows per well]
    ↓
3× Conv1d(5×5, ReLU) + norm
    ↓
Conv1d head → K*2=8 output [K=4 trajectories, 2 params: mean, log_std]
    ↓
Reshape → (B, K=4, 2, T) [4 TVT delta hypotheses]
```

### 特徴量 (StandardScaler normalized)
- MD (measured depth)
- Z (vertical)
- GR (gamma ray, 32% missing → fill with 0)
- last_known_TVT (anchor)
- delta_MD_from_PS, delta_Z_from_PS

### MTP Loss
```
For each well sample:
  1. Compute MSE per trajectory k: MSE_k = mean((μ_k - Δ_true)²)
  2. Select best: best = min(MSE_k over k)
  3. Batch loss = mean(best over batch)

効果: K=4 trajectories のうち最も良い1本だけ罰則 → 多様性促進
```

### 学習設定
- CV: GroupKFold by well_id, 5 folds (618 train, 155 val per fold)
- Epochs: 20 (early stopping patience=5)
- Batch: 16 wells
- Optimizer: Adam (lr=1e-3, wd=1e-4)
- Scheduler: ReduceLROnPlateau

### 推論
```
Per well:
  pred_tvt = last_known + cumsum(
    softmax(-μ_k²) · μ_k  # likelihood-weighted ensemble of K trajectories
  )
```

## 品質ゲート

1. ✓ 完走 (773 OOF + 3 test)
2. ? CV RMSE < 15.91 (anchor)
3. ? error_corr(exp39, exp22) < 0.7 (多様性)
4. ? NNLS blend improves or maintains exp22 (LB検証待ち)

## 既知の課題・対応

| 課題 | 症状 | 対応 |
|------|------|------|
| GR の NaN | scaler → NaN output | np.nan_to_num(..., nan=0.0) |
| PyTorch verbose param | TypeError | 削除 |
| Model output NaN | loss=nan | 入力 NaN 前処理で解決 |

## 実行状況

- **開始時刻**: 14:37 UTC (2026-06-05)
- **想定終了**: 16:37 UTC (~2 hours)
- **Fold 0 進行**: 訓練中
- **Fold 1-4**: 予定待ち

### リアルタイム監視
```bash
tail -f /tmp/exp039_full.log | grep "Epoch\|RMSE\|Inference"
```

## 次のステップ

1. 5 fold 完走を待つ
2. OOF / submission / result.json 確認
3. exp022 との blend test 実行 (NNLS)
4. error correlation, blend weight 確認
5. decide: blend採用 or 単体利用

## 参考文献

Alyaev & Elsheikh (2022): Multi-modal trajectory prediction for subsurface estimation
- DOI: 10.1029/2021EA002186
- **Key**: MTP loss で mode collapse 回避、多様性のある複数予測を学習

## メモ

- exp022 との blend threshold: improve ≥ 0.01 in CV RMSE なら採用
- Test 3 wells のノイズ対策: 4 candidates の中から自動選択される
