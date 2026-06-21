# CLAUDE.md

## Project

ROGII - Wellbore Geology Prediction

## Goal

信頼できるCVをもとに、ROGIIのPublic/Private LBを改善する。

## Language Rule

- 思考内容、仮説、変更内容、実験結果、解釈、次アクションなど、人間向けに言語化する内容は原則として日本語で記録する。
- `ROGII_kaggle_obsidian/` に書く文章も日本語を基本とする。
- コード、ファイル名、列名、設定キー、Kaggle公式用語、ログ出力は必要に応じて英語のまま残してよい。

## Orchestration Rule (厳守・必須)

メインエージェント (= claude main loop) は**オーケストラ**として振る舞う。実装は原則 **worker (Agent tool with `model: "haiku"`)** に委譲する。メインが直接実装するのは「設計・統合・最終判断」のみ。

**必須プロトコル:**

1. **タスク分解**: 実装作業を独立したサブタスクに分解する。
2. **worker起動**: `Agent` tool with `model: "haiku"` (低推論コスト) を使い、明示的なinput/output契約とともに起動する。
3. **並列起動**: 独立したサブタスクは **1メッセージで複数 Agent 呼び出し** を行い並列実行する。
4. **成果物検証 (Check)**: workerが戻ったら必ず次を確認する:
   - 期待ファイルが指定パスに生成されているか
   - 主要な出力指標 (CV/RMSE/LB等) が品質基準を満たすか
   - エラーがログに残っていないか
5. **品質ゲート未達なら再起的修正**: 基準値未満なら **修正指示付きで再度 worker 起動**。修正の余地がなくなるまで自己修正ループを回す。
6. **PDCA継続**: 目標 (LB目標値、CV目標値) 達成まで Plan / Do / Check / Act を止めない。CV改善が止まったら別アプローチに切替えて継続する。
7. **GitHub push / Kaggle submit** は必ずユーザーの明示許可を得る。

**worker起動時の必須記述事項:**
- 目的 (何を達成するか)
- 入力 (どのファイル・データを使うか、絶対パスで)
- 出力 (どこに何を保存するか)
- 品質基準 (CV/RMSE/件数等の数値ゲート)
- 失敗時の対応 (エラーログをどこに残すか)

**メインが直接書いてよいケース (例外):**
- 1〜2行のEdit / コマンド実行
- worker成果物のレビュー・修正指示文の作成
- 戦略判断・blend重み最適化など、コンテキスト集約が必要な統合作業

## Non-negotiable Rules

- `data/raw` は絶対に変更しない
- 成果物、コミット、Obsidianノート、提出物に個人情報・秘密情報・公開リポに含めてはいけないものを入れない
- `.env`、`.kaggle/`、API key、access token、個人の絶対パス、raw Kaggle data、processed parquet、model artifact、OOF、submissionは公開リポに含めない
- commit前に `git status`、staged files、秘密情報スキャンを確認する
- GitHubなど外部リモートへのpushは、必ずユーザーの明示許可を得てから行う
- row random CVは禁止
- 実験は1回につき1目的だけ変更する
- 実験結果は必ず `experiments/expXXX/` に保存する
- OOF、submission、`result.json`、`notes.md`を必ず出力する
- `EXP_SUMMARY.md`を毎回更新する
- `ROGII_kaggle_obsidian/` に思考内容、仮説、変更内容、実験結果、解釈、次アクションを必ず記録する
- CV改善だけで判断せず、リーク懸念をnotesに書く
- 学習時間は原則20分以内。重い実験は事前に明記する
- 既存の良い実験を壊さない
- feature追加時はablationできるようにfeature set名を分ける

## Preferred Workflow

1. `EXP_SUMMARY.md`を読む
2. `ROGII_kaggle_obsidian/00_Index/Home.md`を読む
3. 直近のbest CV / best LB / leak_risk実験を把握する
4. 新しいexp_idを作る
5. configを作る
6. 実験を実行する
7. `result.json` / `oof.csv` / `submission.csv` / `notes.md`を保存する
8. `EXP_SUMMARY.md`を更新する
9. `ROGII_kaggle_obsidian/05_Experiments/` の該当実験ノートを更新する
10. `ROGII_kaggle_obsidian/06_PDCA/Daily_Log/YYYY-MM-DD.md` にPlan/Do/Check/Actを記録する
11. 次の改善案を3つだけ提案する

## Obsidian Knowledge Rules

Obsidian vault: `ROGII_kaggle_obsidian/`

必ず記録するもの:

- コンペ理解・データ理解が更新された内容
- 新しい仮説と、その仮説が生まれた理由
- 実装した変更内容
- 実験config、CV、LB、OOF分析、失敗理由
- リーク懸念、CV-LB gap、採用/棄却判断
- 次にやること

更新先の目安:

- コンペ/評価/ドメイン理解: `01_Competition_Understanding/`
- データ構造/欠損/EDA: `02_Data_Understanding/`
- 特徴量案: `03_Feature_Ideas/`
- CV/評価方針: `04_CV_and_Evaluation/`
- 実験結果: `05_Experiments/`
- 日々のPDCAと意思決定: `06_PDCA/`
- 失敗分析: `07_Error_Analysis/`
- 提出結果: `09_Submissions/`
- データ基盤/ETL/処理設計: `10_Data_Engineering/`

Obsidianには巨大ファイルを置かない。raw data、parquet、model、OOF、submissionはコードリポジトリ内に保存し、Obsidianからパスで参照する。

## Important Concepts

- GroupKFold by well_id
- GroupKFold by typewell_id
- last_known_TVT anchor
- typewell duplicate detection
- OOF error slicing
- CV-LB gap
- tree model blend
