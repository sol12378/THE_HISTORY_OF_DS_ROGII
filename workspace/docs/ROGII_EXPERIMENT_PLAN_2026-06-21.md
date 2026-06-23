# ROGII 実験計画（長期 / 中期 / 短期）

作成: 2026-06-21 / 締切: 2026-08-05（残り約45日）/ 提出: 5回/日

## 0. 現在地（事実確認）

| 項目 | 状態 |
|---|---|
| 自分たちの提出済みbest | **LB 7.625**（exp080 sp45-fleongg clean fork, 2026-06-11） |
| 公開LB7.1ファミリ | exp090=7.168 / exp091=7.201 / exp093=7.295 / exp092・094=7.1付近。**source取得済・未自前提出** |
| 確定下限方針 | `minimum_submission_baseline_lb: 7.1`（これ未満のスコアは提出しない） |
| 我々固有の直交資産 | exp026 PF×geom（単体LB8.672だが公開stackと**誤差相関0.611=部分直交**）/ Group F 幾何外挿 / CUDA(RTX2080S)解禁 |
| 既知の壁 | per-well offset は leak-free信号で予測不能（GR照合/空間/NN/typewell-attn の全paradigmで10回確認）。公開7.1も**この壁を解いていない**（同一情報源のblend+projectionの到達点） |
| LB上位 | 1位 5.986 / 2位 6.228 / 3位 6.487。公開手法の天井 ~7.1–7.4 |

基盤は稼働可（処理済テーブル・leak-safe fold・外部資産・bootstrap 4/4 done）。

## 1. 長期視点（〜8/05・残り45日）— ゴール設定

- **最終目標**: 公開天井(7.1)を確実に超え、**LB 6台に到達**。ストレッチで6.4以下（=銀〜金圏）。
- **質的命題**: 7.1→6台のギャップ(~1ft)は「同一lik-PF/GBMの多様なblend」では埋まらない。
  鍵は **offset壁を別角度で攻める固有技術**（下記Phase4）。
- **撤退ライン**: Phase4が空振りでも、Phase1–3で7.1ファミリ＋我々のPF直交注入により **LB 6.9–7.0** は堅い。最低限ここを確保する。
- **運用規律（research_brief準拠・厳守）**:
  - row-random CV禁止、必ずwell単位hold-out（GroupKFold by well_id, 評価はhidden tail行のみ）
  - train-only地層列(ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA)・typewell Geology は test側生成経路が無い限り使用禁止
  - public overlap leak はパイプライン/LB校正のみ。**モデル選択・賞には転移しない**（exp023/034で確認済）
  - multi-seed・CV-LB整合・leakage分類・submission gate を提出条件にする

## 2. 中期視点（2〜3週間）— LB7.1確保 → 7.0割れ

### M1. LB7.1ファミリの自前再現と確定下限化（最優先・低リスク）
- exp090(lightningv08 7.168)を **intake gate**（leakage分類→ローカルrebuild→submission shape gate→exp080/090比較）に通し、自分たちのkernelとして提出。
- 目的: 確定提出を 7.625 → **7.1台**へ更新し、以降の全blendの新baseにする。
- グループA(exp090/093)・B(exp091/094)・C(exp092 dual)の3系譜で**最良1本を base**に確定。

### M2. 我々のPFを「第4エンジン」として直交注入（最有望）
- 公開blend(7.1)に exp026 PF×geom を加える。著者自身「decorrelated 3rd sourceが唯一の改善路」と明言、実測corr 0.611。
- 実装規律: **借り物OOFは使わず、PFを採点時に自前計算**（exp073の負の転移=借り物OOF膨張を回避）。
  `T = (1-w)*T_public + w*T_exp026` を nested-fold OOFで重み決定し、CV-LB転移を必ず確認。
- 期待: 7.1 → **6.9–7.0**。

### M3. lik-PF同等化＋未活用特徴の移植（中EV）
- PFを single-scale(8.0) から公開同等の **multi-scale{3,5,8,12}+selector** へ（exp040/exp071実装済を統合）。PF成分 11.0→10.3 を狙い blend底上げ。
- iaztec EDA由来の未使用特徴を stack入力として移植: GRオフセット残差曲線 / affine GR較正(a,b) / pfx_rmse(照合品質) / 残り5地層のformation surface。**per-well hard特徴化は過学習するので必ずstack入力として**。

## 3. 短期視点（今週・数日）— 即着手アクション

### S1.（Day1–2）下限確保: exp090 自前提出
- intake gate 4段を実行 → ローカルsubmission生成 → shape gate → 提出。**確定best 7.625→7.1台へ**。
- Go条件: shape OK / leakage low / 公開報告LBと±0.1で再現。

### S2.（Day2–3）CV-LB校正基盤の再確立
- 7.1 base の nested-fold OOFを再構築し、我々のPF/geomとの誤差相関を再実測（exp073の歪み=imputer fold毎非再構築を排除）。
- これが M2/M3 のblend重み決定の土台。

### S3.（Day3–5）PF第4エンジン注入の最小実験
- 自前PFを採点時計算する self-contained kernel に `w` 1パラメータ注入。nested OOFで `w` 決定 → 1提出でCV-LB gap実測。
- Go: CV改善が gap≤1.0 でLBに転移。No-Go: gap拡大 or LB>7.1なら即rollback（負の転移＝exp073教訓）。

### 自律実行の起動
- bootstrap健全性は確認済。S1–S3は `rogii_actions.yaml` にアクション定義 → `rogii_autonomous_loop.py`（または汎用session経路）で実行。
- 各実験は result.json / oof.csv / submission.csv / per-well 診断を必須出力。worst-well sliceを毎回更新し「難井戸にも効くか」を監視。

## 4. Phase4（本命・高難度, 中期後半〜長期）— offset壁の質的突破

EV順（GPU解禁を活用）:
- **4b 学習型emission(GPU)**: PF尤度 P(GR|深度) を生ガウシアン→小型CNN/Transformer学習スコアへ。typewell×lateral局所文脈。5-fold 3分実績。
- **4a constrained DTW**: lateral GR↔typewell GR の制約付き/確率的DTW（本流未使用・我々のexp052素朴DTWは失敗、制約+正則化版は未踏）。
- **4c teacher-forced 枝ランカー(GPU)**: multi-hypothesis枝を回帰でなく分類で「選択」（教師あり枝ランカーは未踏）。

各Phase4実験は単体での壁突破でなく、**PF emission改善→PF全体底上げ→blend反映** の経路で評価する。

## 5. マイルストーン

| 期日目安 | 目標 |
|---|---|
| 〜6/23 | 確定提出 LB7.1台（S1完了） |
| 〜6/30 | PF直交注入で LB6.9–7.0（M2） |
| 〜7/14 | lik-PF同等化＋特徴移植で base強化（M3） |
| 〜7/28 | Phase4 のうち最有望1–2本を実装・評価 |
| 〜8/04 | 最終2提出を選定（best CV-LB整合 + 直交保険blend） |
