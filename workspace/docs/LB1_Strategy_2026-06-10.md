# ROGII LB/PB 1位 (LB 5台) 到達戦略 — 2026-06-10

対象: ROGII Wellbore Geology Prediction (RMSE, code competition, 採点=隠しtest数百well再実行)
現在地: **exp073 nested CV 8.79 (leak-free) 提出実行中** / 確定best LB 8.280 (exp072)
目標: **LB 5台 = 実質1位** (現1位 SaintLouis 5.986、2位 6.228、5台は1チームのみ)

---

## 1. これまでの実験の分析 (exp001〜exp075, 約75実験)

### 1.1 スコア進化と「効いたもの」

| 節目 | CV | LB | 何が効いたか |
|---|---:|---:|---|
| anchor baseline (exp001) | 15.91 | - | 最終既知TVT固定延長 |
| LGB+trajectory+GR (exp008) | 13.81 | 12.339 | 勾配ブースティング+系列統計 |
| **幾何外挿 Group F (exp014)** | 13.53 | - | hidden区間のX/Y/Z/MD完全既知性の活用 |
| モデル多様化blend (exp018-020) | 13.32 | - | LGB/XGB/Cat+NN+attn (+0.2のみ) |
| **Particle Filter (exp022)** | 11.02 | - | **パラダイム転換: 回帰→well単位逐次追跡** |
| PF×geom blend (exp026) | 10.06 | 8.672 | 誤差相関0.426の直交blend |
| artifact stack注入 (exp051) | 9.27 | 8.316 | 公開GBMスタックとのblend |
| projection後処理 (exp072) | 9.086 | 8.280 | U座標robust多項式 |
| **公開資産統合 (exp073)** | **8.69-8.79** | 実行中 | 正直OOF再構築+我々PF(重み0.43)の直交性 |

**パターン**: 大きなジャンプは全て「アーキテクチャの転換」(回帰→追跡、単独→直交blend) で起きており、同一パラダイム内のチューニングは +0.0x〜0.2 しか出ない (exp025 PFチューン -0.04、20革新案ほぼ全滅)。

### 1.2 「効かなかったもの」= leak-freeの壁 (8方向から確認済み)

| 試み | 結果 | 教訓 |
|---|---|---|
| GR直接照合 (exp009/011/013/013b/017) | 全滅 | GR多価性で逆変換不安定。点照合は不可 |
| 空間TVT補間 (diag_spatial, exp058) | RMSE 150-186 / 29.2 | well間隔~6400ftでoffset拘束±185ftのみ |
| per-well offset回帰 (diag_self_calib他) | corr ≤ 0.155 | **offsetはleak-free特徴量で予測不能** |
| NN単体 (exp019/020/039/046/053/075) | 14.6-16.5 | NNはPFの逐次追跡を再現できない |
| leak lookup (exp023/034/044) | LB 10.79/format error | 採点wellはtrainに不在。leak死 |
| 「breakthrough」3件 (self-calib 5.54 / field surface 0.626 / 案19) | 全てleak | 桁違いCVはまずleakを疑う (Decision_Log突合) |

### 1.3 誤差構造の核心 (LB 5台への鍵)

- PF per-well RMSE **中央値 5.66** — 既に1位(5.986)水準。pooled 11.0に押し上げるのは **47本のbroken well (>20ft, 全体の6%)**。
- オラクル診断: 1 offset/well補正で 8.21、**4 offset/wellで 4.39** → 構造上 5台は到達可能。
- broken→good級に直せれば pooled ≈ 6.0 (試算済み)。
- **結論: LB 5台は blend の延長では到達しない。drift well救済 = 追跡ロスト問題の解決が唯一の道。**
- 上位陣の裏付け: fleongg自身が「これ以上はdecorrelatedな第3ソースが必要」と明言し7.572で停止。6.2-6.9帯はGM級(C.Deotte 7.19が中位に沈む難コンペ)。SaintLouis 5.986だけが分離=何か質的に違うことをしている (おそらくdrift well解決)。

---

## 2. 公開情報の分析 (2026-06-10監査)

- **fleongg fle3n-v4 (LB 7.572)**: Engine A (ravaghi GBM stack 0.3 + 128-seed lik-PF selector 0.7 + projection) × Engine B (185特徴LGB 0.6 + likPF 0.4)。全資産公開 (ravaghi artifacts / pilkwang package / koolbox)。
- **重要パラメータ**: lik-PF = 128seed×500粒子, scales(3,5,8,12), init_spread 4.5, 尤度softmax加重。selector = n_eval/z_span 6bin routing。projection deg4 β0.75。warmup tau85。
- **leak判定**: train-TVT lookup分岐は採点時死にコード (exp044で確認)。公開LBは正味性能。
- **我々の優位**: exp026 PF は公開blendと誤差相関 0.611 の直交成分 (exp073で重み0.43=最大)。
- **pilkwang package の罠 (exp073で発見)**: sequence_tcn は OOF (mean +1.6) を採点時推論 (hidden-only cold start, mean -35.6) で再現できない。**「OOFで強い」と「デプロイで再現できる」は別物** — 成分採用時は必ず推論経路の忠実性を検証する。

---

## 3. フェーズ別ロードマップ (LB 5台へ)

ゲイン見積りは CV→LB gap ~0.8-1.4 の実績転移則に基づく。

### Phase 0: exp073 着地確認 (実行中, 〜本日)

- **作業**: kernel完走→submit→LB測定。LB_Tracking更新。
- **期待**: CV 8.79 → **LB 7.3〜7.7** (gap則)。公開best 7.572と同等以上、目標7.419に肉薄〜達成。
- **成功基準**: LB ≤ 7.7。エラー時はログ修正・再push。
- **意義**: 新パイプライン (公開資産+我々PF) のCV→LB転移係数を確定させる。以後の全見積りの較正点。

### Phase 1: drift well 法医学 (1-2日) — 全フェーズの土台

- **作業**:
  1. exp073 blend OOF (8.69) の per-well 誤差センサス: broken (>20ft) / bad (8-20ft) / good の人口と、pooled² への寄与分解。
  2. broken well の故障モード分類: (a) 早期分岐ミス (wrong branch), (b) 後半ドリフト, (c) typewell不適合 (重複/遠方), (d) GR品質不良, (e) 幾何特異 (急傾斜/長尺)。PNG画像レビュー併用。
  3. 各モードの「もしオラクルで直したら」pooled改善の定量化 → 投資優先度を数値で決める。
- **成功基準**: 故障モード別の well リストと寄与表。修正可能性ランク。
- **見込み**: 直接ゲイン0だが、Phase 2-4の照準。**ここを飛ばすと過去の20案全滅を繰り返す。**

### Phase 2: トラッカー大改修 — multi-hypothesis化 (3-5日, 本命)

oracle 4.39 (4-offset/well) が示す通り、勝負は「追跡ロスト時に正しい枝へ戻れるか」。

- **2a. lik-PF移植+selector統合**: 公開の尤度加重multi-scale PF (彼ら) と我々のexp071 selector (PF 10.91→10.47実績) を統合。コードは入手済み。見込み: PF単体 11.0→10.3-10.5。
- **2b. multi-hypothesis tracking (MHT)**: PFの単峰collapse が broken の主因と仮説。複数offset枝を**最後まで枝刈りせず並走**させ、well末端で (i) 累積尤度、(ii) 空間prior (近傍well formation surface との整合)、(iii) 幾何滑らかさ、で遅延選択。これは「4-offset oracle」の実装形。
- **2c. 再初期化 (re-anchoring)**: 尤度崩壊検知時に、空間priorからoffset候補を再サンプル。exp045 (NCC emission) は失敗したが、raw-GR emission + 空間prior再初期化は未踏。
- **2d. 区分的DP/HMM**: 状態=typewell深度offset の離散HMM をViterbiでなく forward-backward+複数経路で。beam (exp021) の±2制約を撤廃した全域版。
- **成功基準**: broken 47本 → 25本以下 (leak-free LOWO)。pooled PF 11.0 → 9.5以下。blend CV 8.69 → **7.8-8.2** → LB ~6.6-7.2。

### Phase 3: CUDA GPU 活用 — NN を「予測器」でなく「審判」に (2-4日, Phase 2と並行可)

過去のNN 6連敗は全て「NNにTVTを直接予測させた」。GPUが解禁された今、役割を変える:

- **3a. 学習型emission model**: PFの尤度関数 P(GR | typewell深度) を、生GRガウシアンから小型CNN/Transformerの学習スコアに置換。typewell×lateral GRの局所文脈を見る。学習はGPUで数分 (exp075実績: 5-fold 2.9分)。
- **3b. hypothesis ranker**: Phase 2bのMHT枝 (well毎に4-16本) を、系列NN (cross-attention: lateral GR ↔ typewell profile ↔ 近傍well) でランキング。「offsetを当てる」(不可能と証明済み) のではなく「候補列から選ぶ」(分類問題、教師=正解枝) に変換。**これが本コンペにおけるNNの正しい使い方の最有力仮説。**
- **3c. drift検知器**: broken well を予測時に検知する分類器 (過去AUC 0.5で失敗したが、multi-scale尤度・MHT枝分散など新特徴で再挑戦)。検知できれば per-well fallback routing が解禁。
- **成功基準**: 3bで枝選択精度 > 累積尤度選択。blend CV -0.3以上。
- **リスク**: NN系は6連敗の実績。だが全て「回帰」での敗北であり、「選択/採点」は未踏。GPUで試行コストが激減 (1実験3分) したため、ダメでも損失小。

### Phase 4: 空間prior の追跡器への注入 (2-3日)

- ravaghi特徴の FormationPlaneKNN / DenseANCC (leak-free, 公開実証済み) は現在GBM経由でしか効いていない。これを**PF/MHTの状態prior・尤度項として直接注入** (例: 粒子初期分布を近傍well surface 由来に、伝播中も弱い引力項)。
- exp056の教訓: 空間surfaceは self除外で 29.2 と単体では弱いが、**±10-20ft の事前分布としては十分** (PFのinit_spread 4-4.5に対して情報量がある)。
- **成功基準**: broken well の早期分岐ミス型が減る。pooled -0.2以上。

### Phase 5: 統合・ルーティング・後処理最適化 (1-2日)

- Phase 2-4 の新成分を含めた joint nested NNLS 再構築。
- well-regime別ルーティング (selector拡張): n_eval/z_span/GR品質/MHT枝分散でbin分けし、bin毎に blend重みを切替 (fold一貫性ゲート必須)。
- projection/warmup/平滑の再チューン (現状 d4 β0.75 / tau85 / w101)。
- **成功基準**: CV ≤ 7.0 → **LB 5.8-6.2 圏**。

### Phase 6: LB検証・Private防衛 (継続)

- 提出は1日2回を規律的に: (1) 各Phaseの着地確認、(2) LB probing はしない (公開勢の轍)。
- CV-LB gap が崩れたら即停止して原因分析 (新成分の hidden-test 非忠実性を疑う — pilk_tcn の教訓)。
- 最終2提出: (a) 安全板 = exp073系 (検証済みgap)、(b) 攻撃版 = drift救済フル。
- Private対策: fold一貫性・重み安定性ゲートを全採用判断に適用済みを維持。LBチューニング成分 (公開blendの0.55等) は使わない。

---

## 4. 数値ロードマップ (転移則 LB ≈ CV − 1.0±0.3)

| マイルストン | CV | 期待LB | 状態 |
|---|---:|---:|---|
| exp072 (確定) | 9.086 | 8.280 (実測) | done |
| **exp073 (Phase 0)** | 8.69-8.79 | **7.3-7.7** | 実行中 |
| Phase 2 (tracker改修) | 7.8-8.2 | 6.6-7.2 | 計画 |
| Phase 3+4 (NN審判+空間prior) | 7.0-7.6 | 5.9-6.6 | 計画 |
| Phase 5 (統合) | ≤ 7.0 | **5.8-6.2** | 計画 |
| 理論限界 (4-offset oracle) | 4.39 | ~3.5-4.5 | 参考 |

LB 5台には CV ~6.5-7.0 が必要。oracle 4.39 との差は十分あり、**構造上は可能**。律速は drift well 救済率のみ。

## 5. 原則 (過去の事故からの不変則)

1. **breakthrough級CVはまずleakを疑う** — Decision_Log棄却履歴と突合 (exp056事故)。
2. **OOFの強さ≠デプロイ忠実性** — 新成分は必ず「採点時推論経路」で検証 (pilk_tcn事故)。
3. **1実験1変更・fold一貫性ゲート・well単位GroupKFold** を崩さない。
4. **offsetの直接予測に投資しない** (8方向で不可能確認済み)。投資先は「追跡の頑健化」と「候補からの選択」。
5. worker成果物のleak自己検証をmainが必ず再確認する。
