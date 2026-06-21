# Decision Log

| Date | Decision | Reason | Evidence | Link |
|---|---|---|---|---|
| 2026-05-25 | Obsidianをプロジェクト記憶として使う | 多数の実験をまたいで判断理由を残す必要がある | User request | [[Home]] |
| 2026-05-25 | leak lookupと本物のCVを分離する | test wellsがtrainに存在する | local data check | [[Leakage_and_Risks]] |
| 2026-05-25 | modeling前にbase data contractsを固定する | コンペ中のschema変更はCV比較を壊す | data engineering planning | [[Long_Term_Data_Engineering_Plan]] |
| 2026-05-25 | 最初の本命baselineはanchor差分LightGBMにする | anchorが強く、差分学習の改善量を測るのが安全で速い | exp001 RMSE 15.909853、exp003 RMSE 15.054865 | [[exp003_lgb_anchor_trajectory]] |
| 2026-05-30 | Group E（typewell GR alignment）を完全封印する | importance上位だが汎化に有害。リーク除去後はbaselineを+0.276悪化。中域corr多数派が劣化 | exp009b CV悪化、exp010 corr層別、exp011 leak DiD +0.153 | [[exp011_typewell_leak_test]] |
| 2026-05-30 | best維持はexp008 (CV=13.808621)。次は非typewell軸で攻める | Group E系列が全滅。trajectory post-PS / anchor guard等の別軸へ | exp009/009b/011 全てexp008に劣後 | [[exp008_gr_rolling]] |
| 2026-05-30 | typewell由来特徴量を使うならtypewell-grouped foldを必須化する | 34 wells(13グループ)が同一typewell共有、well_id GroupKFoldでリーク | typewell signature重複検出、exp011 leak確定 | [[exp011_typewell_leak_test]] |
| 2026-05-30 | CVドリブン開発方針を確定する | exp008でCV改善1.246→LB改善1.808確認。CVがLBに確実転化。gap=-1.469 | submission_001 LB=12.339 | [[submission_001_exp008]] |
| 2026-05-31 | **GR-typewell直接照合をNO-GO判定**。Tier2の本丸仮説を棄却 | leak-free直接照合(exp013/013b)が全変種でexp008に劣後。狭窓±10+exp008中心でも最善+0.017悪化、純match+0.6〜2.6。GR多価で逆変換不安定。exp009特徴量化も含め3実験一貫で失敗 | exp013 A=43.08/anchor15.91、exp013b best13.826>exp008 | [[exp013_gr_match_go_nogo]] [[exp013b_gr_local_refine]] |
| 2026-05-31 | Tier2投資をGRアライメント以外へ振り替える | GR照合NO-GO。残路はDTW系列照合のみだが高リスク。代わりにモデル多様化(XGB/Cat)・系列平滑性・幾何特徴(Group F)へ | Phase1判定結果 | [[Strategy_2026-05-31]] |
| 2026-05-31 | anchor guard(exp012)は本採用しない(保留) | +0.054だがfold非一貫(fold0 -0.077)。864通りsweepのOOF過学習兆候。strong_guard α=0.85の頑健部分のみ将来検討 | exp012 fold別delta | [[exp012_anchor_guard_exp008]] |
| 2026-06-02 | **CV<=5 は GR含む正規手段で到達不可能と確定**。oracleで上限を定量化 | geom形状は正しく残差は低周波offset。理想offset(1/well)でも上限8.21、4/wellで4.39。だがGRが選ぶoffsetは真offsetとcorr=+0.155と極弱。GR特徴量化(exp017)もleak-safe typewell-foldで-0.067悪化。GR照合は全形態(exp009/011/013/013b/017+oracle)でNO-GO | diag_gr_ceiling, diag_seq_align, exp017 | [[exp017_gr_align_features]] [[gr-offset-ceiling]] |
| 2026-06-02 | 現実的次手はモデル多様化(XGB/CatBoost blend)。現best維持=exp015(13.520) | GR路は完全閉鎖。geom形状が良い構造的知見を活かしつつ、誤差非相関でCV底上げを狙う | oracle診断+exp017 | [[Strategy_2026-05-31]] |
| 2026-06-02 | per-well offsetは leak-free信号で予測不能と確定(self-calib corr=-0.028) | known自己ホールドアウトのgeom外挿biasは真hidden offsetと無相関。GR(0.155)も含めoffsetは復元不能。これがLB首位6.8の天井理由 | diag_self_calib | [[gr-offset-ceiling]] |
| 2026-06-03 | **「GR照合 全形態NO-GO」を部分撤回**。well単位確率的ハードトラッカー(PF)は有効 | 参照3notebook解読→PF(128seed×500粒子,状態pos=TVT+Z,GR尤度)を移植。**pooled CV 11.024=全手法最良**(geom13.53/blend13.32/anchor15.91超)。完全leak-free。3 test well平均≈5.0で参照"4.71ft"再現。並行作業のNO-GOは学習的/ソフト/グローバル形態のみで、PFは未検証だった | exp022_particle_filter CV=11.024 | [[exp022_particle_filter]] |
| 2026-06-03 | Beam Search(決定論)は不採用。PFを採用 | beam CV15.70≈anchor(±2運動制約が粗く迷子)。PFは連続状態+尤度加重で曖昧性を統合し11.02 | exp021 vs exp022 | [[exp021_beam_track]] |
| 2026-06-03 | **新本線=PFをensembleに統合**。PF×geom blend で現時点 CV 10.16 | PFとgeom誤差相関0.426と低い。0.6·PF+0.4·geom+平滑=10.16(前best13.32から+3.16,全fold一貫9.49〜11.07)。次はtree/NN含む多段blend+壊れwellゲート+PF調整+Kaggle kernelでLB転移検証 | PF×geom blend分析 | [[exp022_particle_filter]] |
| 2026-06-03 | リーク(exp023)は提出採用しない方針(賞転移しない見込み) | 3 test wellはtrainに完全TVTありRMSE=0だが、LB#1=6.8≠0→train TVT≠Kaggle採点真値。privateで無効の公算。正直CVと厳密分離して保持 | exp023 leak_rmse=0 vs LB#1=6.8 | [[exp023_leak_lookup]] |
| 2026-06-03 | **多段ブレンド(exp024)を新best採用 CV=10.077**。PFをensembleに統合 | NNLS(delta,非負,nested5-fold)で pf/geom/trees/nn/attn をブレンド→10.090、+平滑10.077。前best13.32から+3.24、全fold一貫(9.28〜11.08)。重み=PF0.66+attn0.47(幾何系は相関0.98-1.0で冗長、PFのみ0.43直交)。leak-free | exp024 nested CV=10.0897→平滑10.0769 | [[exp024_multistage_blend]] |
| 2026-06-03 | 壊れwellゲートは不成立。NNLS重みで頑健化する | 47壊れwell(pf_rmse>20)オラクル上限→attn=9.46だが、leak-free検出シグナル(PF自己GR残差corr0.225)で分離不能、gate適用は悪化(10.09)。PF信頼性は予測不能(diag_self_calibと整合) | gate検証 | [[exp024_multistage_blend]] |
| 2026-06-03 | PFチューニング採用(init_spread=4,PN=0.01)。full再実行→再ブレンド予定 | ~129well subsetでbaseline10.726→ispr4_pn0.01=10.468(−0.258)。広いinit spread+高め運動雑音の相互作用で探索性向上 | exp025_pf_tune subset grid | [[exp025_pf_tune]] |
| 2026-06-05 | **Horizon distillation (A/B/C案) 全棄却**。情報理論的に(MD,X,Y,Z,GR)→horizonsの関数等価で寄与なし | exp028b: horizons予測R²=0.945だがTVT予測+0.088悪化(CV14.60→14.69)。Niccoli証言「per-formation classifier無効」と完全整合 | exp028b | [[exp028b_horizon_resid]] |
| 2026-06-05 | **Multi-typewell PF採用(blend寄与)**。単体は悪化だがexp022と相補的 | exp030b: 空間最近傍3typewellでPF並走、単体CV=12.15で-1.12悪化、しかしexp033 NNLS blendで weight=0.12 採用 | exp030b | [[exp030b_multi_tw_vec]] |
| 2026-06-05 | **Physical Likelihood PF採用(blend寄与)**。normalized GR + GR derivative で47壊れwell中23救出 | exp031: 単体CV=20.38だが**誤差相関 vs exp022 = 0.333と低い**(相補的)。直接2-blendでCV10.40 | exp031 | [[exp031_pf_physical_lik]] |
| 2026-06-05 | **exp033 = 新best CV 9.977** (-0.085 vs exp026)。8 components NNLS blend | pf_orig0.31 + attn0.29 + pf_tuned0.21 が主軸、残りは多様化補助。test3wells RMSE 3.99/4.62/5.64 | exp033 | [[exp033_final_blend]] |
| 2026-06-05 | **1位LB6.5は leak-free到達不能 と最終確定**。leaders are sub-9 RMSE | per-well offsetはleak-free信号で予測不能(corr<=0.155)、47壊れwell大半解けない。我々のLB予想8.5は「sub-9 leader圏」 | Niccoli証言 + 全実験 | [[exp033_final_blend]] |
| 2026-06-02 | **exp018 モデル多様化blendを採用、新best CV=13.340426**(blend+exp015平滑化) | LGBM+XGB+CatBoost等加重blendが+0.184。平滑化で+0.0012、全fold一貫、leak-free。exp015(13.520)から+0.180 | exp018 CV=13.3416→平滑13.3404 | [[exp018_model_blend]] |
| 2026-06-02 | 多井戸空間モデル(diag_spatial)を棄却 | TVT~f(X,Y,Z)はhidden RMSE149-186。well間隔~6400でTVTは±185しか拘束できずanchor±16に完敗。wellが疎すぎてoffset回収不能 | diag_spatial | [[gr-offset-ceiling]] |
| 2026-06-02 | **1位LB(6.8)接近の本命=typewell-aware NN(exp020)を検証→offset回収不能を最終確認** | lateral GR↔typewell profileのcross-attention(微分可能DTW)でもNN単体16.5と悪化。GR照合/空間/NN/attentionの全paradigmでoffset取れず。public LB6.8は3本分散要因が大きい | exp020 NN16.5/blend13.322 | [[exp020_typewell_attn]] |
| 2026-06-02 | **現best CV=13.320964**(exp018+exp019+exp020 3-blend+平滑)。次は実LB提出でgap測定 | 全fold一貫leak-free。exp015(13.520)から+0.199。グローバルCVでの大幅改善は情報限界で頭打ち | exp020 3-blend平滑 | [[exp020_typewell_attn]] |
| 2026-06-06 | **leak路線 完全終了**。採点test=train非存在のhidden well群と確定 | exp044純leak提出が"submission incorrect format"エラー=hidden wellはtrainに無くleak lookup全NaN。exp034/036/037が全10.794だったのも整合(leak不適用、壊れ4-comp model支配) | exp044 format error | [[exp034_hybrid_leak]] |
| 2026-06-06 | **SaintLouis 5.986はleak-free確定**。leak仮説を完全棄却 | 構造的leak(3 test wellがtrain存在)は採点対象外のサンプルwellのみ。実採点はhidden well | exp044 | [[exp034_hybrid_leak]] |
| 2026-06-06 | **5.986への律速=broken well 47本のPF追跡ロスト**と特定 | PF per-well RMSE中央値5.66(既に5.986並)。47 broken(>20)がpooled CV 11.02に押上げ。broken→good修正でpooled概算6.0。4-offset/well oracle 4.39が裏付け。offsetはGR逐次追跡で可(中央値5.7)、特徴回帰では不能(corr-0.028) | per_well分布 + diag_gr_ceiling | [[exp022_particle_filter]] |
| 2026-06-06 | **P0-P4はexp026未超え**。exp026(LB8.672)が唯一の検証済best維持 | exp040 multi-scale PF CV10.979(+0.045)、exp041 residual GBDT CV10.53だがexp026の10.06に劣る。multi-tw/physical-lik(全体適用)は有害確定 | exp040/041, pdca_blend | [[exp026_final_blend]] |
| 2026-06-06 | **次戦略=broken well救済に全集中**。blend/特徴でなくPF追跡頑健性が勝負 | P-A診断→P-B再初期化/P-C Beam hybrid/P-D観測物理化(broken限定)。各LB検証(gap1.4転移実証済) | 戦略策定 | [[exp022_particle_filter]] |
| 2026-06-06 | **Stage0/B-1 (drift不変NCC emission) 失敗確定**。raw GRが識別信号と判明 | exp045 subset(broken15+good15): broken救済1/15, good全15本悪化(median +81ft)。NCCの振幅不変性がGR絶対値の識別力を破壊→PF発散。外部レポートの「emission健全化が前提」仮説はこのデータで誤り。exp022のraw GR点値照合が既に94%well最適 | exp045 subset_check | [[exp022_particle_filter]] |
| 2026-06-06 | **leak-free天井=exp026 LB8.672 がほぼ限界と認識**。全方向で壁確認 | leak死/外部データ粗い/broken識別AUC0.5/NCC emission壊滅/P0-P4未超え。1-offset oracle 8.21に肉薄。blend(geom)が既にbroken mitigation。残る賭け=artifact blend(安全)/MTP-CNN(本命だが不確実) | 全実験 | [[exp026_final_blend]] |
| 2026-06-06 | **(A) artifact OOF blendが本物の前進**。nested CV 10.0→9.21(-0.80, 全fold一貫) | thbdh5765/rogii-v11-fresh-artifacts(GBDT meta-stack, 195特徴=pf/beam/sc/formation/spatial)のOOF予測を入手・行アライン検証OK(再構成CV 10.503一致)。我々PFと誤差相関0.59で部分直交。NNLS artifact重み0.54。予想LB~7.8(楽観)〜8.3(kojimar実績準拠) | artifact OOF blend nested | [[exp026_final_blend]] |
| 2026-06-06 | **(B) MTP-CNN(exp046)はネガティブ**。CV10.74, blend寄与なし | NN系(exp019/020/039/046)全敗。PFの逐次追跡をNNは再現できず | exp046 | [[exp046_mtp_cnn]] |
| 2026-06-06 | **artifact単独再現は非現実的**(195特徴pipeline無し)。提出はkojimar notebook fork経由 | models.pkl+特徴名のみ、特徴生成コード無し。kojimar helper(特徴pipeline)は403。提出するならkojimar公開notebook fork+我々PF注入 | inference_config/features.json | [[exp026_final_blend]] |
| 2026-06-06 | **TabICLはM5 GPU(MPS)で稼働不可**。CPU専用 | device='mps'で RuntimeError(!self.is_mps INTERNAL ASSERT, TensorShape.cpp:1414, MPS未実装op)。CPUは可だが763k context burn-in激遅。ただしv11(LGB/Cat)はTabICL不要でCPU完結→本筋に影響なし | scripts/test_tabicl_mps.py | [[exp049]] |
| 2026-06-06 | **v11 artifact ローカルCPU実行成功=提出経路確立**。helper不要 | 公開v11-infer notebook(TabICL無し)をローカル適応、3 test well予測生成(train-truth RMSE3.83、in-sample寄りで楽観だがpipeline正常)。honest signal=OOF nested CV 9.21(artifact+PF+geom) | exp049 | [[exp026_final_blend]] |
| 2026-06-07 | **exp051 LB 8.316 = 新best**(exp026 8.672から-0.356)。artifact blend転移成功 | 0.438*v11_artifact(GBDT meta-stack) + 0.562*exp026(PFxgeom)。nested CV 9.27→LB 8.316(gap 0.95)。kojimar公開blend8.293とほぼ同水準=公開手法の到達上限に到達。leak-free | exp051提出 | [[exp026_final_blend]] |
| 2026-06-07 | **革新1-4 全てLB改善なし**。exp051 8.316が引き続きbest | 1 DTW CV23.3(emission不足) / 2 neural CV16.2 blend寄与0(corr0.68) / 3 self-calib 5.54はLEAK(known_mask=~last_known_TVT.isna()が全行True→hidden真値でfit、leak-free=11.02改善ゼロ、diag_self_calibと整合) / 4 PNG=CSV冗長で非有用。self-calib 5.54は提出すればPrivate崩壊、棄却 | exp052-055 | [[exp051]] |
| 2026-06-07 | **チャネルB/A/C 全てleak-freeで改善なし**(うちB/Cはworker leak混入を検出) | B(構造曲面)worker版0.626は**self込みleak**(現wellのASTNU点がKDTreeに残存→自分予測)。正しいLOWO(self除外)=pooled29.2/median12.3でPF11.02に負け(ただしdiag_spatial150より大幅良=密lateral標本は有効)。A(Geology marker)CV1217失敗。C(空間offset)mode aは7.10だがhidden使用leak、mode b空間平滑は無改善。**3手とも本物の前進なし** | exp056/057/058 | [[exp051]] |
| 2026-06-07 | **本セッションの全"breakthrough"はleakだった**(self-calib5.54, field-surface0.626)。leak-free検証(LOWO/正しいknown mask)が必須と再確認。検証済みbest=exp051 LB8.316 | worker成果は必ずleak自己検証。per-well offsetはleak-free予測不能の壁(7回確認) | 全leak検出 | [[exp051]] |
| 2026-06-07 | **20革新案 全てleak-free改善なし**(4件目leak検出: 案19) | 案19 conformal 8.79は pf_rmse/anchor_rmse(真値必要)でfallback判定=オラクルleak。案2 逆方向 -0.03(微small real)。群III(7-10)実装失敗122-403、群V(15-17)121-126(offset転移不可再確認)、案3 cycle corr0.001。群II(4-6)/群IV(11-14)はworker脱線で未実行。本物の前進ゼロ | exp060-065 | [[exp051]] |
| 2026-06-07 | **20革新案 完全検証完了: 本物の前進ゼロ**。exp051 8.316確定best | 唯一の生存案=群II構造曲面(PFと直交corr0.037, surf+PF NNLS 10.07)だが、artifactと相関0.325で冗長→exp051に加えても9.27→9.25(−0.02ノイズ級)。群IV信号変換=GR局所多価性で全滅(全表現±30ft局在化、生GR最良)、群V=offset転移不可(121-126)、群III=PF局所最適で改造全滅、案19=オラクルleak。壁(per-well offset leak-free予測不能)を8方向から再確認 | exp060-067 | [[exp051]] |
| 2026-06-07 | **projection(構造座標robust多項式)が genuine leak-free改善**。nested 9.27→9.086 | exp068: proj_U_deg5/7=9.17(−0.10), 生TVT平滑(proj_raw)は9.67悪化→U=pred+Z−anchor座標が必須。全部入り(artifact+exp026+surface+residual+pf)+proj_U=9.086(−0.19, 予想LB~8.1)。ramp(tau)無効、PF600/150悪化、apply_pp w_pf=0(PF混合無効)。selector workerは未完+fold-train RMSE7.1異常→main再実験 | exp068/070 | [[exp051]] |
| 2026-06-08 | **selector(main再実験)=genuine改善**。PF 10.91→10.47(−0.44, leak-free) | worker(exp069)の7.1はバグ/leak。exp071厳密版: per-scale PF + regime(n_eval,z_span)6bin、variant(scale/hold)選択をfold内train RMSEのみ(leak-free)。我々の実装。blend価値は要評価 | exp071 | [[exp051]] |
| 2026-06-08 | **exp072(projection版)提出**(ref53462017)。projection=我々の実装でleak-free | exp051 blend + 構造座標U robust多項式。nested 9.27→9.086。LB反映待ち | exp072 | [[exp068]] |
# 2026-06-10: LB 7.419 以下狙いの主戦略 (改訂)

- **訂正**: 当初「exp056 field surface (CV 0.626) の提出kernel化」を主戦略としたが、これは**2026-06-07に確定済みのself-leak** (cKDTreeに評価well自身のASTNU点が残存、正しいLOWO=29.2でPF以下) を見落とした誤った計画だった。即日棄却。戦略立案時はDecision_Logの棄却履歴との突合を必須とする。
- 検証済み現状: **exp072 = LB 8.280 (現best)** (ref 53496203) / nested CV 9.086。目標7.419まで-0.861。
- 新決定: 主戦略は **exp073 公開資産統合**。公開notebook監査 (5本) の結果、fleongg fle3n-v4 (LB 7.572) は全て公開データセット (ravaghi artifacts / fleongg models) で構成され、GroupKFold OOF内蔵のTrainer pickleから正直なOOFを構築できる。我々のexp026 (PF×geom) は彼らに無い直交成分であり、nested NNLS blendでCV 9.086更新を狙う。
- リーク判断: 彼らのtrain TVT lookup分岐は採点時に発火しない死にコード (exp044の知見と整合)。public LB 7.572は正味性能で信頼可。ただしblend重み・selector閾値はLB probing由来の可能性があるため自前OOFで再決定する。
- 詳細: [[Strategy_2026-06-10_LB7419_plan]]

# 2026-06-10: exp073 公開資産統合で新best CV 8.690 (leak-free, -0.396)

- **結果**: 公開資産(ravaghi artifacts / pilkwang package)の正直OOFと我々の直交成分(exp026 PF×geom, exp075b TCN)をnested NNLS blend + 後処理(projection deg4 β0.75 + mean101 + warmup85) → **nested CV 8.690**(現best exp072 9.086から−0.396、全fold一貫改善8.16〜9.52)。
- **重み**: pf 0.428(我々PF=最大重み)+ pilk_tcn 0.255 + rav_lgb3 0.251 + pilk_cat 0.154 + tcn_resid 0.112。公開stack(10.04)と我々stack(10.03)の誤差相関0.611=直交が源泉。著者の「改善には非相関な第3ソースが必要」を我々のPFが実証。
- **leak検証**: PF OOF 10.984 = exp025と完全一致(再現性確認)。外部OOFはtrainer.overall_scoreと一致(GroupKFold by well, self-well除外)。全成分well単位OOF。
- **subset別CV**: ours-only 9.78 / ours+rav 8.98 / **ours+pilk 8.76** / joint 8.69。pilkwangパッケージ単体が最強外部。
- **CUDA活用**: exp075 GPU-TCN(RTX 2080 SUPER, torch cu124)。v1直接14.87→residual版(exp075b)13.39。blend寄与は小(0.11)だが正。NN単体は弱い従来知見を再確認しつつ、residual化で初めて微寄与。
- **提出方針**: pilkwang notebook(rav+pilk全成分を採点時計算)をfork + 我々exp026(pf+geom)注入 + joint重み + projection。我々TCN_residはtorch同梱回避でドロップ(8.697)。目標LB<=7.419。CV→LB転移: 公開pipeline CV9.21→LB7.572、我々は8.69でそれを上回る → LB 7.0〜7.4圏を期待。
- 詳細: [[exp073_public_assets_integration]]

# 2026-06-10: exp073 LB 8.630 = 負の転移で棄却。best=exp072 8.280維持

- **結果**: exp073 (公開資産blend, nested CV 8.79) を提出 → **LB 8.630**。CV改善(9.086→8.79)にもかかわらず**LB悪化(8.280→8.630)**。CV-LB gapがexp072の+0.81から+0.16へ縮小=favorable transfer消失。
- **原因**: 外部OOF(ravaghi/pilkwang)が pilkwang manifest明記の「OOF imputerをfold毎に再構築しない」により**楽観的に膨張**していた。我々のPF/geom/v11/projectionは「LB<CV」のfavorable transferを持つが、膨張した外部成分(重み~0.66)がそれを希釈・破壊。
- **判断**: exp073を棄却。**確定best LB = exp072 8.280**。今後の成分採用は「我々が推論経路まで管理」+「favorable transfer実績あり」に限定。借り物外部OOFは不採用。
- **方向転換トリガー**: loop#1(routing棄却)+loop#2(MHT棄却)+exp073(LB悪化)=3連続非改善。blend/stacking exhausted、GR尤度ベースのoffset-tracking walled(8回確認)。残る未踏=**非尤度の枝選択(空間formation prior=幾何方向)**。
- 詳細: [[loop_log]] (docs/)
