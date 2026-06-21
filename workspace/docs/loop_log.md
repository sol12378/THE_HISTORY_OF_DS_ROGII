# 自律実験ループ ログ (目標: LB Top 5 / LB 5台)

各行: ループ# | 手法 | CV | LB | 採否 | 次の一手

## ── 新ループ (2026-06-11, LB5計画ベース) ──
実験カウント: 5ごとに有効性精査、10ごとにLB影響調査。base=sp45 fork(採点中 ~7.4)。

| exp# | 手法 | 結果 | 採否 | 次の一手 |
|---|---|---|---|---|
| 1 | Phase B: two-filter PF smoother(逆分散結合, subset257) | smoother 25.06 vs fwd 16.35 | **棄却(生は破綻)** | 後向きPFも壁=偽枝に低分散確信。一致ゲートで救済試行 |
| 2 | 一致ゲート smoother(\|sm-fwd\|<=T, offline) | **gated T15=16.02 vs fwd 16.35(-0.33)** | **採用(救済成功)** | 全773で検証→PF底上げ |
| 3 | full-773 gated smoother(64seed, gate T12) | **PF 10.984→10.796(-0.19)** | **採用(clean leak-free PF底上げ)** | oof_smoother_gated.csv保存。Phase A blend土台 |

**★ fork LB着地 = 7.625 = 新best!**(従来8.280から-0.655)。sp45-fleongg fork(clean、公開datasetのみ)。これが新baseで、以降の全実験の較正点。Phase A2(fork + 我々PF直交blend, w≈0.3)へ。



| # | 日付 | 手法(1変更) | CV | LB | 採否 | 次の一手 |
|---|---|---|---:|---:|---|---|
| 0 | 06-10 | exp073 公開資産統合blend (pf+geom+ravaghi+pilk_cat+proj) | 8.79 | **8.630** | **棄却(LB悪化, best=exp072 8.280)** | 外部OOF膨張で負の転移。方向転換 |
| 1 | 06-10 | Phase1 forensics + per-well routing予測可能性テスト | 8.69(分析) | - | 棄却(routing単純版8.36止まり) | Phase2: tracker改修(lik-PF+selector) |
| 2 | 06-10 | MHT multi-restart PF (offset{-40..40}尤度選択, subset207) | 22.73 vs single17.98 | - | **棄却(悪化-4.7, 悪化53本)** | LB着地待ち→較正後に枝選択の非尤度信号(空間prior/NN ranker)を検討 |
| 3 | 06-10 | **方向転換(幾何)**: 空間formation surface offsetで枝選択できるか診断 | surf 835/corr-0.135 | - | **棄却(corr負, 枝ランク不能)** | offset壁を尤度+空間の両面で確定(9回目)。後処理(proven-transfer)へ |

---

## ★ 重大: exp073 LB 8.630 = 負の転移 (2026-06-10)

| 実験 | CV | LB | gap(CV-LB) |
|---|---:|---:|---:|
| exp072 (確定best) | 9.086 | **8.280** | +0.81 (LBがCVより良い) |
| exp073 (公開資産blend) | 8.79 | 8.630 | +0.16 (転移消失) |

- **CV改善(9.086→8.79)なのにLB悪化(8.280→8.630)**。確定best LBは **exp072 8.280** のまま。
- 原因: 外部OOF(rav/pilk)が「imputerをfold毎に再構築しない」(pilkwang manifest明記)で**楽観的に膨張**。我々のPF/geom/v11/projectionが持つfavorable transfer(gap+0.8)を希釈。
- **教訓再確認**: OOFの強さ≠デプロイ性能(pilk_tcn事故と同根)。**借り物の外部OOFはCV-LB転移を破壊しうる**。今後の採用は「我々が推論経路まで管理できる成分」+「favorable transfer実績のある成分」に限定。
- loop#1/2 + exp073 = 3連続非改善 → **方向転換**(blend/stacking exhausted、offset-tracking walled)。残る未踏=非尤度の枝選択(空間prior=幾何方向)。

---

## ★ 中間総括 (2026-06-10, loop#3後): offsetの壁を多面確定

確定best = **exp072 LB 8.280**(top ~3%, leak-free)。LB1位 SaintLouis 5.986 とは2.3ft差。

**per-well offset(=どの地層枝にいるか)は leak-free信号で予測不能** を、独立した複数角度で確認(計9回):
- GR直接照合 (exp009/011/013/013b/017): 全滅
- GR尤度PF枝選択 (loop#2 MHT): corr事実上0、悪化
- 空間formation surface枝選択 (loop#3): corr **-0.135**(逆相関)
- self-calibration (diag_self_calib): corr -0.028
- 空間TVT補間 (diag_spatial): well疎すぎ
- NN回帰 (exp019/020/039/046/053/075): 全敗
- 外部GBM stack OOF (exp073): CV良→LB悪化(膨張)

**含意**: LB 5台は「offsetを当てる/枝を選ぶ」の改良では到達しない(情報が無い)。残る可能性:
1. **後処理(proven-transfer)**: projection系の精緻化。歴史的に+0.0xだが唯一positive transfer。
2. SaintLouis 5.986 が使う未知技術(別データ/別定式化)の発見 — 公開情報に手掛かり乏しい。
3. broken/bad wellの根本原因(データ品質・typewell対応)の個別調査 — 高コスト低確度。

**方針**: 闇雲な offset実験は停止(EV低)。loop#4以降は (1)後処理 に絞り、proven base(exp026/exp072)上でCV→LB転移を確認しながら小幅改善を積む。それも頭打ちなら、新規公開notebook/discussionの再監査で技術発見を待つ。

---

## ★★ ループ一旦停止 (2026-06-10, loop#5後): 最終構成と総括

### 確定 best submission (再現手順)
- **exp072 = LB 8.280** (Public, top ~3% / ~63位相当)。
- 構成: `0.438 * v11_artifact(GBM meta-stack, thbdh5765/rogii-v11-fresh-artifacts) + 0.562 * exp026(PF×geom自己完結)` + U座標robust多項式 projection(deg5)。
- kernel: `kaggle_notebooks/exp072_proj/rogii_exp072.py`(base64埋め込み自己完結)。提出はKaggle UIで notebook を "Submit to Competition"(CSV直接提出は400=notebook-only comp)。
- leak-free。CV(nested 9.086)→LB 8.280、gap +0.81(LBがCVより良いfavorable transfer)。

### なぜ LB 5台に届かないか(6ループ+多数の過去実験の結論)
- **per-well offset(どの地層枝にいるか)は leak-free信号で予測不能**。約10通りで確認: GR照合/GR尤度PF/空間surface(corr-0.135)/self-calib(-0.028)/空間補間/NN回帰6種/外部GBM stack/学習的per-well再重み付け(過学習)。
- **固定重みblendが学習的per-well重み付けに勝つ**(loop#5: 10.05 vs 10.80)= 利用できるper-well構造が存在しない。
- 外部公開に LB5台の手法なし(公開最高9.25)。借り物外部OOFはCV-LB転移を破壊(exp073: CV8.79→LB8.630)。
- broken/bad well(誤差の78%)は wrong-branch / partial drift だが、枝を正す情報が leak-free データに無い。理論オラクル(4-offset/well=4.39)は到達可能性を示すが、その offset を選ぶ手段が無い。

### 停止判断
6連続非改善・全探索軸(model/blend/geometry/外部/後処理近傍/reweighting)が頭打ち。闇雲な継続は資源浪費のため**自律ループを一旦停止**。

### 再開する価値がある条件(LB5台への現実的な道)
1. **新しい公開技術の出現**: コンペ終盤に上位がnotebook公開したら即監査(romantamrazov系の更新等)。
2. **broken well根本原因のデータ調査**(高コスト): 47 broken wellのtypewell対応・GR品質・PNG画像を個別精査し、データ品質起因なら除外/補正ルール、地質ブロック起因なら専用typewell再選定。
3. **後処理の専用探索**(小幅・proven-transfer): exp072 base上で projection/平滑の体系的グリッド。期待+0.0x。
4. Private shake-up待ち(我々はleak-free・fold一貫・重み安定でPrivate耐性は高い設計)。

| 4 | 06-10 | 外部技術偵察(web/公開notebook) | - | - | 棄却(公開最高9.25、LB5台手法は非公開) | loop#5=我々成分のper-well学習的再重み付け(disagreement信号) |
| 5 | 06-10 | per-well学習的再重み付け(GBMメタスタッカー, pf+geom+leak-free特徴) | 10.80 vs exp026 10.05 | - | **棄却(過学習で悪化-0.76)** | 固定重みが勝つ=利用可能なper-well構造なし。ループ一旦停止(壁確定) |

## Loop 4 (外部偵察)
- web/Kaggle検索: 公開notebookは全て9+級(DWT 9.251 / hill climbing / better 9.956)。SaintLouis 5.986級の手法は**非公開**。突破の外部手掛かり無し。
- Act: 外部に答えは無い。残る最有望=**我々の転移成分のper-well学習的再重み付け**。loop#1のdisagreement信号(誤差corr+0.25=全特徴で最強)+ routing oracle 7.09 を狙う。借り物OOF不使用で転移は正直。次loop#5で実装。

---

## Loop 1 詳細 (Phase 1 forensics + routing test)

### Plan (誤差分析→仮説)
exp073 blend OOF (8.69) のper-well誤差センサスで投資先を特定する。仮説: 既存3成分{blend,pf,geom}のper-well routingで大きな余地があるか?

### Do (1変更: 既存OOFのper-well分析、提出なし)
- forensics: 帯別人口とpooled²寄与を分解。
- routing test: leak-free per-well特徴(disagreement|pf-geom|, z_span等)が「どの成分を信頼すべきか」を予測できるか。

### Check (結果)
- **帯別寄与**: good<8=566well(22%), **bad 8-20=188well(45%, 最大)**, broken>20=19well(33%)。blendは既にPFのbroken 48→19に削減済み。**最大の誤差源はbad帯(8-20ft)= 部分ドリフトwell群**。
- broken 19本は全て大bias(±18-43ft)= wrong-branch lock。長尺・大z_span。
- **per-well oracle pick min(blend,pf,geom)=7.094**(blend 8.69から-1.6の余地)。
- pf<blend on 292well(38%), geom<blend on 153well(20%) → per-well最適混合は大きく変動。
- leak-free信号: disagreement(|pf-geom|) が誤差と spearman **+0.25**(最強)だが、単純なdisagreement-gated geom-routingは pooled 8.36止まり(oracle 7.09に遠い)。geom_better_rateは高disagree帯でも16-27%と低い。

### Act (採否と次)
- **単純routing/reweight = 採用見送り**。理由: 既存3成分は同じドリフトを共有するため、選択の予測可能ceilingが低い(過去のoffset予測不能の壁と整合)。
- **次の本命 = Phase 2 トラッカー改修**。bad+broken帯(78%の誤差)を減らすには「より良い追跡」が必要。最小変更の第一歩 = 公開のmulti-scale lik-PF(128seed×4scale尤度加重)+ 我々のexp071 selector(PF 10.91→10.47実績)を移植し、PF単体 11.0→10.3-10.5 を狙う。これがblend底上げの土台。
- CV-LB gap監視: exp073 LB着地後に較正(現状exp072 gap 0.806)。

### 提出メモ
- exp073 kernel (sol12378/rogii-exp073-blend) commit完走。**CSV直接提出は400(notebook-only comp)**。Kaggle UIで "Submit to Competition" が必要。

---

## Loop 2 詳細 (MHT multi-restart PF)

### Plan
broken 19本は全てwrong-branch lock(大bias)。現PF init_spread 4ftでは遠方の正枝に届かない。offset{-40,-20,0,+20,+40}から並走し、well毎に総尤度最大の枝を選択(4-offset oracle 4.39の実装形)。subset(rmse_blend>=8の207well)で48seed×5offsetで概念検証。

### Do
exp076_mht_pf.py。pf_singleにinit_off追加。restart毎にmax single-seed total loglikをスコアとして枝選択。

### Check
- single(off0) 17.98 → **MHT 22.73 (悪化 -4.7ft)**。rescue 3本のみ、悪化53本。
- 選択offset分布: 0.0=115, ±20=55, ±40=37 → 非ゼロを92本で選んだがその多くが誤り。
- CV-LB gap: 該当なし(subset診断、未提出)。

### Act
- **棄却**。GR尤度は枝をランクできない(多価性で偽ピーク選択)。「offset leak-free予測不能(corr<=0.155)」の8回目の確認。
- 含意: Phase2bのMHTが前提とした「累積尤度による枝選択」は死亡。枝選択には**非尤度信号**(空間formation prior / NN ranker / 幾何滑らかさ)が必須。
- 次: **まずexp073のLB着地(ref53533960 PENDING)を待ち、8.79の転移を較正**してから、非尤度の枝選択(Phase4空間prior)を検討。LB次第で方向確定。
