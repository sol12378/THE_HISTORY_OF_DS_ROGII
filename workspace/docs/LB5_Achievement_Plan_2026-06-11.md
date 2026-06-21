# LB 5台達成 統一マスター計画 — 段階別詳細分解

作成: 2026-06-11（`LB5_Achievement_Plan` + `LB1_Roadmap_from_7.5` を統合）
前提: **sp45-fleongg fork (sol12378/rogii-sp45-fleongg-fork) 提出済み・採点中**（ref 53552794, 推定LB ~7.4-7.5）
目標: **LB 5台（1位 SaintLouis 5.986 超え）= 実質1位**

> 本書は2つの先行計画を統合した唯一のマスター計画。出典は `docs/LB1_Strategy_2026-06-10.md` / `docs/LB1_Roadmap_from_7.5_2026-06-10.md` / `docs/loop_log.md` / `ROGII_kaggle_obsidian/`。

---

## 0. エグゼクティブサマリ

- **現在地**: 確定best exp072 LB 8.280 → **sp45-fleongg fork で公開7.4系を再現・提出中**（新base）。
- **ギャップ**: 7.4 → 5.986 = **約1.5 ft**。公開8 notebookは全て同一系譜で、この1.5ftを埋める技術を**誰も公開していない**。
- **律速**: per-well の「構造追跡ロスト」(bad帯188本の部分ドリフト + broken 47本のbranch lock)。offsetを当てる直接手法は**10回死亡確認済み**。
- **唯一の現実的経路**: (i) 我々のexp026 PF(直交corr 0.611)のblend注入 → 7.0圏【高確度】、(ii) **系列大域最適化(DTW/smoother)＋学習型emission(GPU)** で構造追跡を底上げ → 5-6台【本命・成功率主観30-50%】。
- **正直な見通し**: Phase A-B で **7.0圏(上位10位前後)は高確度**。**5台はPhase C×Dの成否次第**。

---

## 1. 現状の正確な棚卸し

### 1.1 LB状況（出典付き）
- 1位 SaintLouis **5.986** / 2位 6.228 / 3位 6.487（実測 leaderboard 2026-06-10）。
- 公開手法の天井 ~**7.4-7.57**（fleongg fle3n-v4 = 7.572）。LB5台の手法は**非公開**（loop#4: 公開notebook最高9.25級）。
- 我々の確定best: exp072 LB **8.280**（top ~3%）。**exp073(公開資産blend)はLB 8.630で棄却**(負の転移)。

### 1.2 スコア進化と「効いた転換」(exp001〜075)
| 節目 | CV | LB | 効いた要因 |
|---|---:|---:|---|
| anchor (exp001) | 15.91 | - | last_known_TVT固定延長 |
| LGB+traj+GR (exp008) | 13.81 | 12.339 | GBM+系列統計 |
| 幾何外挿 GroupF (exp014) | 13.53 | - | hidden区間のX/Y/Z/MD既知性 |
| **Particle Filter (exp022)** | 11.02 | - | **回帰→well単位逐次追跡(転換)** |
| PF×geom blend (exp026) | 10.06 | 8.672 | 誤差相関0.426の直交blend |
| artifact注入 (exp051) | 9.27 | 8.316 | 公開GBM stack blend |
| projection (exp072) | 9.086 | **8.280** | U座標robust多項式 |
| sp45 fork (exp080) | - | ~7.4(採点中) | **公開7.4系の忠実再現** |

**法則**: 大躍進は全て「アーキテクチャ転換」(回帰→追跡、単独→直交blend、自前→公開pipeline再現)。同一paradigm内チューニングは +0.0x〜0.2 のみ。

### 1.3 公開7.4パイプラインの解剖（8 notebook監査の統合）
全7.4-7.57系は **fleongg fle3n-v4 が源流**の同一系譜:
```
T_sp45  = projection_deg4_β0.75( 0.30*ridge_stack + 0.70*likPF_selector )
T_final = 0.55*T_sp45 + 0.45*T_fleongg(pretrained LGB + warmup tau85 + SG61)
          [+ model-package gated ±0.5%]  [+ guarded overlap override = Public限定leak/Private no-op]
```
| 要素 | 実体 | 我々の保有 |
|---|---|---|
| ravaghi ridge stack | LGB×3+CB×2 (185特徴) のRidge meta (α1.66) | ✓ artifact+OOF |
| **lik-PF selector** | 128seed×500粒子, scale{3,5,8,12}, init_spread4.5, n_eval/z_span 6bin routing | ✗(我々はsingle-scale) |
| sp45 projection | dU=TVT+Z-anchor robust deg4多項式 β0.75 | ✓ exp068(deg5) |
| fleongg Engine B | 185特徴LGB pretrained + warmup85 + SG61 | ✓ artifact |
| pilkwang TCN/GBM | drift_ncc model-package | △ TCNは採点時再現不能(下記) |
| guarded override | train-TVT lookup | **不採用**(Private no-op) |

監査ノート: `ROGII_kaggle_obsidian/05_Experiments/exp_audit_public_notebooks_2026-06-10.md`

### 1.4 per-well誤差構造（攻撃対象）
exp022 per_well + exp073 forensics の層別:
- **good帯 (<8ft): 566本** — PF追跡成功(最良0.58ft、中央値5.66)
- **bad帯 (8-20ft): 188本** — 部分ドリフト = **最大誤差源(pooled²の45%)**
- **broken (>20ft): 19-47本** — wrong-branch lock(±18-43ft bias、全て長尺+大Z_span)
- per-well oracle min(blend,pf,geom) = **7.094**(現有成分の選択だけで-1.6ft余地だが選択信号無し)
- **4-offset/well oracle = 4.39 / 1-offset oracle = 8.21** → 構造上5台は到達可能、律速は「追跡ロスト→正枝復帰」のみ。

**結論**: bad帯188の部分ドリフト + broken 47のbranch lockは**逐次フィルタの構造的弱点**(一度迷子になると戻れない)。これが唯一の攻撃対象。

### 1.5 offsetの壁（10回の実証 = 投資禁止領域）
per-well offset(地層枝)は leak-free信号で直接予測不能:

| # | 手法 | 結果 |
|---|---|---|
| 1 | GR直接照合 (exp013/013b) | 43.08 / NO-GO |
| 2 | MHT尤度枝選択 (loop#2) | 22.73 vs 17.98(-4.7悪化) |
| 3 | 空間surface offset (loop#3) | corr **-0.135**(逆相関) |
| 4 | self-calibration | corr -0.028 |
| 5 | 空間TVT補間 | hidden RMSE 149-186(疎すぎ) |
| 6 | NN回帰6種(exp019/020/039/046/053/075) | 全て10.7-16.5でPF未満 |
| 7 | 外部GBM stack OOF (exp073) | CV改善でもLB悪化(負の転移) |
| 8 | 学習的per-well再重み付け (loop#5) | 10.80 vs 固定10.05(過学習) |
| 9 | GR特徴量化 leak-safe (exp017) | -0.067悪化 |
| 10 | oracle診断 | 1-offset理想でも8.21が天井 |

**含意**: 「offsetを当てる」全形態死亡。残る道は (i)直交blend、(ii)**系列照合・大域最適化・学習型尤度による間接的な追跡改善**のみ。

### 1.6 我々の固有資産
1. **exp026 PF×geom** (CV10.06/LB8.672): 公開stackと誤差相関**0.611**=部分直交。公開勢が持たない decorrelated 3rd source。
2. **geom幾何外挿**(GroupF): hidden区間の完全既知幾何を使う構造傾斜投影。
3. **CUDA GPU**(RTX 2080 SUPER): TCN 5-fold 2.9分実績(exp075)。学習型emissionの実機学習可。
4. favorable CV-LB transfer(exp072 gap+0.81)。**ただし借り物OOFで破壊される**(exp073)。

---

## 2. 技術的根拠（外部文献調査 2026-06-11）

### 2.1 大域最適化(smoother/Viterbi/DTW) > 逐次フィルタ
- SMC理論: forward-backward particle smoother は filterのparticle degeneracyを後退再帰で補正(Klaas+ ICML2006)。**事後一括予測(本コンペ=test末端までGR観測あり)では全データ使用のsmoother/Viterbiがfilterに理論的優位**。
- geosteering直接適用: Alyaev+2021 — 確率的解釈+DPが専門家29人中28人を上回る。SPE Journal2025 "Geosteering Robot"=PF多重解釈+RL。
- DTW log correlation: Lineman+1987以来の古典。Wheeler&Hale2014, Sylvester2023(査読)が大域相関を実証。
- 制約付き>無制約: Sakoe-Chiba帯(Geler+2016/19)、学習制約(Ratanamahatana&Keogh SDM2004)。
- **exp052 DTW失敗(CV23.3)の整合**: exp052は無制約・未較正・素朴照合(Decision_Log)。文献が有効性を示すのは「制約+較正+大域DP」で**未踏**。

### 2.2 学習型emission(最強の外部根拠)
- **DHGC(JMSE/MDPI 2025, 査読)**: 1D CNNを177,026 triplet(北海8坑井GR)で学習→スコアNN+DP整列。**未見89ペアで相関0.35→0.91、成功率98.9%、素DTW比8.2倍速**。本コンペ同一設定(GR系列の坑井間照合)で「学習+DP ≫ 生信号DTW」を定量実証。
- 補強: ASDNet(2023), CMT-Hiformer(2025), Le+2019(CNN depth matching)。

### 2.3 リスクの根拠
- Group E封印(exp011)は「点単位特徴量化」のleak。学習emissionは「系列整列コスト」の別形態だが、**fold内train wellsのみ学習**の規律必須。
- exp073教訓: 採点時に推論経路を管理できない成分はfavorable transferを破壊。**全新成分はkernel内自己完結計算**で設計。
- pilkwang TCNの罠: OOF(mean+1.6)を採点時hidden-only推論(mean-35.6)で再現不能。**「OOFで強い」≠「デプロイで再現可」**。

---

## 3. 数値ターゲットの逆算

CV-LB転移則: LB ≈ CV − 0.8〜1.4(自前成分のみで成立)。
**LB 5.9 には自前管理パイプラインで pooled CV ≈ 6.7〜7.3 が必要**。
クリーンCV現状: exp072系 9.09。→ **クリーンCVをあと約2ft削る**のが課題の定量的定義。

---

## 4. 段階別 達成計画（Go/No-Go ゲート付き）

> 原則: 1 Phase = 1機構。各Phaseは pooled CV ゲート **+ per-well軌跡プロット**(bad帯/broken回復本数)で Go/No-Go。CV-LB gapを毎提出監視。

### Phase A（即時〜2日）: 7.5 base確保 + PF直交注入【防衛線・高確度】
- **目的**: 公開7.4を自分のbestにし、我々のPFで直交ゲインを得る。
- **A1**: sp45 fork の LB確認(採点中, ref53552794)。**ゲート: LB ≤ 7.6**。確認後 best更新・LB_Tracking記録。
- **A2**: exp026 PF(corr0.611)を fork kernel内で**自己完結計算**し、nested OOFで重み決定して `T=(1-w)*T_public + w*T_exp026` を注入。借り物OOF不使用(exp073回避)。
- **根拠**: fleongg著者「decorrelated 3rd sourceが唯一の改善路」明言 + corr0.611実測。
- **期待**: 7.4 → **7.0-7.2**(同等強度・corr0.61 blendの理論値 -0.2〜0.4)。
- **ゲートA**: CV-LB gap +0.5以上のfavorable維持。崩れたら巻き戻し。

### Phase B（2-4日）: forward-backward PF smoothing【低コスト先行】
- **目的**: bad帯188本の「途中ドリフト」を将来観測で修正(filteringの構造的弱点を直す)。
- **根拠**: §2.1。test区間も末端までGRありなので smoothing は完全 leak-free。
- **実装**: exp022 PFに backward pass(two-filter smoother / FFBSi簡易版)追加。既存粒子・seed資産を再利用。
- **定量ゲートB**: pooled CV 10.06 → **9.5以下** かつ bad帯188本 median改善 ≥1.0ft かつ broken悪化なし。per-well軌跡でドリフト消失を目視。
- **未達時**: 大域化の効果薄と判断しPhase C期待値を下方修正(DTWは別機構なので中止せず)。

### Phase C（4-8日）: 制約付きDTW / Viterbi 大域アライメント【本命1】
- **目的**: broken 47のbranch lockを大域最適化で解く(逐次でなく全域でtypewell対応を決める)。
- **設計(exp052失敗との差分を明示)**:
  1. 状態空間=(MD, typewellオフセット)、exp022の pos=TVT+Z 分離を維持(素朴照合に戻さない)
  2. 遷移コスト=dip rate物理制約(傾斜変化上限、層厚単調性)
  3. 観測コスト=**affine較正済みGR残差**(iaztec監査の a,b 較正を前段に)
  4. 帯制約=**PF posteriorをSakoe-Chiba帯として使用**(既存資産合成、帯幅依存問題を回避)
- **定量ゲートC**: pooled CV ≤9.0、3 test相当well悪化ゼロ、broken 47のうち ≥10本が<20ftに回復。
- **期待**: blend後 CV 8.5前後 → LB 6.8-7.2圏。

### Phase D（5-10日, Cと並行可）: 学習型emission【本命2・GPU】
- **目的**: PF/DTWの観測尤度を生GRガウシアンから学習スコアへ(構造照合の質を底上げ)。
- **根拠**: §2.2 DHGC(同一設定で0.35→0.91)。
- **設計**:
  1. 教師: train正解TVTから (lateral窓, typewell窓) の正/負対応ペア生成、triplet/contrastive(DHGC方式、**fold内trainのみ**)
  2. モデル: 小型1D CNN(GPU 5-fold数分、exp075で実行可能性確認済)
  3. 統合: 学習スコアを (a)PF/smootherのemissionに置換 (b)Phase CのDTW観測コストに使用 — **C×Dの合成が最終形**
- **定量ゲートD**: 対応判定AUC ≥0.85(DHGC0.997だが大データ、保守設定)、emission置換後PF CV 10.06→9.5以下。
- **リスク低減**: 過去NN6敗は「TVT直接回帰」。本Phaseは「対応判定の2値学習」=別タスク。回帰でなく照合学習である点が決定的に異なる。

### Phase E（中EV・Aの後いつでも）: iaztec未活用特徴で stack強化
- **目的**: 公開stackにあり我々geom/PFが未使用の特徴を移植し、stack/blendを底上げ。
- **候補(iaztec EDA監査由来)**:
  - **GRオフセット残差(tda/tdbc/tdsc/tdpf XX)**: 現推定±offsetのGR-typewell残差曲線=局所matching景観。低コスト有望。
  - **affine GR較正(a,b)**: well間GR感度差をknown区間で1次補正(Phase Cの前段としても使用)。
  - **pfx_rmse(GR照合品質)**: well-levelでPF信頼度→**stack入力として**(hard routingは過学習なので柔らかく)。
  - **6地層×segment別B-value**: 我々はANCC 1層のみ。ASTNU/ASTNL/EGFDU/EGFDL/BUDAの5層surfaceを追加。
- **ゲートE**: stack CV -0.2以上、かつ採点転移確認(exp073教訓: OOF改善だけで採用しない)。

### Phase F（中EV・並行可）: lik-PF + selector の自前同等化
- **目的**: 我々のPF(single-scale8.0)を公開lik-PF(multi-scale{3,5,8,12}+selector)と同等化。
- **資産**: exp040(multiscale 10.98)/exp071(selector 10.47)が実装済。統合。
- **期待**: PF成分 11.0→10.3。Phase A/B/Cの土台強化。

### Phase G（継続）: 統合 + Private防衛
- B/C/D/E/F生存成分 + Phase A blend を nested NNLS統合。fold一貫性・重み安定ゲート。
- **最終2提出**: 安全板(Phase A、7.x確実) + 攻撃版(C×D統合、5-6台狙い)。
- guarded override等Public限定leak**不採用**(exp044で採点時死にコード確定)。
- Private対策: LBチューニング成分(公開0.55等)は自前CVで決め直し、fold一貫性・重み安定を全採用判断に。

---

## 5. マイルストン表と決定木

| Phase | 内容 | 期待LB | 根拠の強さ | 期間 | 依存 |
|---|---|---:|---|---|---|
| A | 7.5 base + PF直交注入 | 7.0-7.2 | 高(corr0.611+著者明言) | 〜2日 | fork LB確認 |
| B | PF smoothing | CV-0.5検証 | 中(SMC理論) | 2-4日 | A |
| C | 制約付きDTW/Viterbi | 6.8-7.2 | 中(文献多数, exp052失敗歴) | 4-8日 | E(較正)推奨 |
| D | 学習型emission(GPU) | C×Dで5.8-6.5 | 中-高(DHGC査読実証) | 5-10日 | — |
| E | iaztec特徴 stack強化 | -0.2級 | 中 | 2-3日 | A |
| F | lik-PF自前同等化 | PF底上げ | 中 | 2-3日 | — |
| G | 統合+Private防衛 | 最終 | — | 継続 | 全Phase |

**決定木**:
1. fork LB ≤7.6 → Phase A2(PF注入)。>7.6なら fork再検証 or fleongg v4 forkに切替。
2. Phase A で7.0圏到達 → B/E/F を低コスト並行。
3. **核心判定**: Phase B(smoother)が CV-0.5達成 かつ Phase D(emission)が AUC0.85達成 → **C×D合成に全力**(5台の本線)。
4. C×D が各ゲート未達 → **5台は現知見で到達不能と判定**。6.8圏防衛(A+B+E+F)+ PNG画像CNN(exp055再挑戦=最後の未踏)に切替。

**正直な確率評価**: Phase A-B+E+F で **6.8-7.0圏(上位10位前後)は高確度**。**5台はC×Dの成否次第、成功率 主観30-50%**。根拠: (i)1位との1.5ftギャップは公開技術に存在しない、(ii)残る未踏は系列大域化+学習照合のみ、(iii)その両方に査読済みの肯定的外部実証がある — 消去法と文献の両面から導かれる現時点で最も期待値の高い経路。

---

## 6. 撤退基準・やらないこと
- **offset直接予測(回帰/分類/尤度選択/空間補間)には一切投資しない**(§1.5、10回の壁)。
- blend重み・selector・projectionの追加チューニングは Phase A 完了で打ち止め(期待値合計-0.3未満、構造的に5台に届かない)。
- 借り物外部OOFのblend禁止(exp073: CV8.79→LB8.630)。全成分kernel内自己完結。
- per-well hard routing/学習reweight禁止(loop#1/5の過学習)。stack入力として柔らかく使う。
- B/C/D全て各ゲート未達なら 5台到達不能と判定、6.8圏防衛+PNG CNNへ。

## 7. 実験規律（鉄則）
1. 借り物OOF禁止、全成分kernel内自己完結(exp073)
2. per-well hard routing/学習reweight禁止(loop#1/5)
3. offset直接予測禁止(10回の壁)
4. 学習emissionは fold内train wellsのみ学習(Group E再発防止)
5. 各Phaseは数値ゲートで Go/No-Go、pooled CV + **per-well軌跡プロット**(bad帯/broken回復本数)で判定
6. CV-LB gap毎提出監視、+0.5未満に縮んだら成分巻き戻し
7. breakthrough級CVはまずleakを疑う(Decision_Log突合、exp056事故)
8. 1実験1目的、experiments/expXXX/ に result.json/oof/notes、EXP_SUMMARY.md更新
9. GitHub push / Kaggle submit はユーザー明示許可を得る
