# LB 1位 (≤5.986) への詳細ロードマップ — 7.5 baseから

作成: 2026-06-10 / 前提: sp45-fleongg fork (推定LB ~7.4) を新baseとして提出。
LB状況: 1位 SaintLouis 5.986 / 2位 6.228 / 3位 6.487 / ... 公開手法の天井 ~7.4-7.57。
**7.4 → 5.986 のギャップ = 約1.4 ft。これは公開手法には無い質的に異なる技術が必要。**

---

## 0. 公開7.5パイプラインの解剖(8 notebook監査の統合)

全ての7.4-7.57系は同一系譜(fleongg fle3n-v4が源流)で、以下の積み重ね:

```
T_sp45 = projection_deg4_b0.75( 0.30*ridge_stack + 0.70*likPF_selector )
T_final = 0.55*T_sp45 + 0.45*T_fleongg(pretrained LGB + warmup + SG61)
        [+ model-package gated ±0.5%]  [+ guarded overlap override = Public限定leak]
```

構成要素と入手元:
| 要素 | 実体 | 我々の保有 |
|---|---|---|
| ravaghi ridge stack | LGB×3+CB×2 GBM stack (185特徴) のRidge meta | ✓ artifact + OOF |
| lik-PF selector | 128seed×500粒子, scale{3,5,8,12}, n_eval/z_span 6bin routing | ✗ (我々はsingle-scale PF) |
| sp45 projection | dU=TVT+Z-anchor の robust deg4多項式, β0.75 | ✓ (exp068, deg5) |
| fleongg Engine B | 185特徴LGB pretrained + warmup tau85 + SG61 | ✓ artifact |
| pilkwang TCN/GBM | drift_ncc model-package | △ (TCNは採点時再現不能) |
| guarded override | train-TVT lookup | Public限定leak, Private no-op → **不採用** |

**重要**: これらは全て同じlik-PF/GBMを共有し、offsetの壁(per-well offset予測不能, 我々が10回確認)を**解いていない**。7.4はあくまで「同一情報源の多様なblend+projection」の到達点。

---

## 我々の固有資産(公開勢が持たない)

1. **exp026 PF×geom** (単独LB 8.672): 公開stackと**誤差相関 0.611**(exp073で実測)= 部分直交。これが「decorrelatedな第3ソース」(fleongg著者が改善の唯一の道と明言)。
2. **geom幾何外挿** (Group F): hidden区間のX/Y/Z/MD完全既知性を使う構造傾斜投影。
3. CUDA GPU解禁(RTX 2080 SUPER): 系列NN/学習型emissionが実機学習可能に。

---

## Phase別ロードマップ

### Phase 0 (本日, 実行中): 7.5 base確立
- sp45-fleongg fork 提出 → LB ~7.4 確認。**現best 8.280 を大幅更新**。
- 成功基準: LB ≤ 7.6。これが以降の全実験のbase。

### Phase 1 (1-2日): 我々のPFを第4エンジンとして直交注入 【最有望・低リスク】
- 仮説: 公開blend(7.4)に exp026 PF(corr 0.611)を加える。著者自身「decorrelated 3rd sourceが唯一の改善路」と明言。
- 実装: forkしたkernelに exp026(pf+geom自己完結)を追加計算 → `T = (1-w)*T_public + w*T_exp026` を nested OOFで重み決定。**借り物OOFでなく我々のPFを採点時計算**(exp073の失敗=借り物OOF膨張を回避)。
- 期待: 7.4 → **7.0-7.2** (corr 0.61, 両者~同等強度のblendで-0.2〜0.4)。
- リスク: 低(proven base + 我々のproven成分)。CV-LB転移を必ず確認。

### Phase 2 (2-3日): iaztec EDA由来の未活用特徴で stack強化 【中EV】
我々のgeom/PFがまだ使っていない、公開stackにある特徴を移植:
- **GRオフセット残差 (tda/tdbc/tdsc/tdpf XX)**: 現推定位置±offsetでのGR-typewell残差曲線 = 局所matching景観の特徴量化。低コスト・有望。
- **affine GR較正 (a,b)**: well間GR感度差をknown区間で1次補正 → PF尤度の質改善。
- **pfx_rmse (GR照合品質)**: well-levelで「PFを信頼できるか」の指標 → blend重みの動的調整(ただしloop#5でper-well学習は過学習したので慎重に、hard特徴でなくstack入力として)。
- **6地層×segment別B-value**: 我々はANCC 1層のみ。残り5層(ASTNU/ASTNL/EGFDU/EGFDL/BUDA)のformation surface特徴を追加。
- 期待: stack RMSE -0.2〜0.5(ただし採点転移は要検証、exp073教訓)。

### Phase 3 (3-5日): lik-PF + selector の自前実装 【中EV, 公開と同等化】
- 我々のPFはsingle-scale(8.0)。公開は multi-scale{3,5,8,12} + n_eval/z_span selector。
- exp040(multiscale 10.98)/exp071(selector 10.47)は実装済み → これらを統合し公開lik-PFと同等のPFを得る。
- 期待: PF成分 11.0→10.3。blend底上げ。

### Phase 4 (本命だが高難度, 5-7日): 7.4→6台の質的飛躍 【高EV高リスク】
公開天井7.4を超えるには、offsetの壁を別角度で攻める必要がある。候補(EV順):
- **4a. DTW系列アライメント**: pilkwang super-stackにconstrained/stochastic DTWあり(本流では未使用)。lateral GR ↔ typewell GR を動的時間伸縮で対応付け、offsetを系列照合で決める。我々のexp052 DTWは失敗したが、constrained版(±制約+正則化)は未踏。
- **4b. 学習型emission (GPU)**: PFの尤度 P(GR|深度) を生ガウシアンから小型CNN/Transformerの学習スコアに置換。typewell×lateral局所文脈を見る。GPU 5-fold 3分(exp075実績)。emission改善はPF全体を底上げ。
- **4c. hypothesis ranker (GPU)**: multi-hypothesis枝をNNで「選択」(回帰でなく分類)。loop#2のMHTは尤度選択で失敗したが、教師あり枝ランカーは未踏。
- **4d. PNG画像活用**: 各wellの.png(stratigraphic display)をCNNで。exp055失敗だが、GPU+typewell対応の再挑戦。高コスト。
- 注意: offsetの壁(10回確認)があるため、4a-4dは「offset/枝を当てる」のでなく「系列照合・文脈学習で間接的に絞る」設計に限る。

### Phase 5 (継続): 多ソースアンサンブル + Private防衛
- Phase1-4の生存成分を nested NNLSで統合。fold一貫性・重み安定ゲート。
- guarded override等のPublic限定leakは**不採用**(Private/賞で無効)。
- 最終2提出: 安全板(7.x確実) + 攻撃版(6台狙い)。

---

## 数値見通し(CV-LB転移則 LB≈CV−1.0±0.3, ただし借り物OOFは転移崩壊に注意)

| マイルストン | 期待LB | 信頼度 |
|---|---:|---|
| Phase 0 sp45 fork | ~7.4 | 高(提出中) |
| Phase 1 +我々PF | 7.0-7.2 | 中-高 |
| Phase 2 +未活用特徴 | 6.8-7.1 | 中 |
| Phase 3 +lik-PF同等化 | 6.7-7.0 | 中 |
| Phase 4 質的飛躍 | 5.8-6.5 | 低(本命だが高難度) |
| 1位 SaintLouis | 5.986 | — |

**正直な評価**: Phase 1-3 で **6.7-7.0 (上位5-15位圏)** は現実的。**LB 5台/1位には Phase 4 の質的飛躍が必須**で、これは公開に無い技術の発見を要する(7.4→6の1.4ftギャップ)。SaintLouisが何をしているかは不明だが、DTW系列照合(4a)か学習型emission(4b)が最有力仮説。

---

## 鉄則(過去の事故から)
1. 借り物外部OOFをそのままblendしない(exp073: CV8.79→LB8.630、膨張で転移崩壊)。採点時に我々が推論経路を管理できる成分のみ。
2. per-well hard routing/学習的reweightは過学習(loop#1/5)。stack入力として柔らかく使う。
3. offset直接予測に投資しない(10回壁確認)。系列照合・文脈・直交blendで攻める。
4. CV-LB gapを毎提出で監視。Public限定leak(override)は賞で無効、不採用。
5. fold一貫性・重み安定ゲートを全採用判断に(Private防衛)。
