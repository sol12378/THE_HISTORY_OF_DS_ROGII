# AGENTS.md

あなたは ROGII Kaggle コンペの実験支援エージェントです。

## 言語ルール

- 思考内容、仮説、変更内容、実験結果、解釈、次アクションなど、人間向けに言語化する内容は原則として日本語で記録する。
- コード、ファイル名、列名、設定キー、Kaggleの公式用語、ログ出力は必要に応じて英語のまま残してよい。
- Obsidian に追記する文章も日本語を基本とする。

## 常に守ること

- 実験は一度に一目的だけ変更する。
- `data/raw` は絶対に変更しない。
- 成果物、コミット、Obsidianノート、提出物に個人情報・秘密情報・公開リポに含めてはいけないものを入れない。
- `.env`、`.kaggle/`、API key、access token、個人の絶対パス、raw Kaggle data、processed parquet、model artifact、OOF、submissionは公開リポに含めない。
- すべての成果物は `experiments/expXXX/` に保存する。
- `EXP_SUMMARY.md` を更新する。
- コンペ理解、実験結果、特徴量案、CV方針、データエンジニアリング判断、提出、次アクションが変わったら `ROGII_kaggle_obsidian/` を更新する。
- 原則20分以内で回る速い実験を優先する。
- Public LBだけを最適化しない。
- リーク懸念を必ず説明する。

## コーディング前

- `EXP_SUMMARY.md` を読む。
- `ROGII_kaggle_obsidian/00_Index/Home.md` があれば読む。
- 直近のbest CV / best LB / leak_risk実験を確認する。
- 実験仮説を明確に書く。
- 必要に応じて仮説を `03_Feature_Ideas/`、`04_CV_and_Evaluation/`、`10_Data_Engineering/` のObsidianノートへリンクする。

## コーディング後

- `result.json` を保存する。
- `oof.csv` を保存する。
- 必要なら `submission.csv` を保存する。
- `notes.md` を書く。
- `ROGII_kaggle_obsidian/05_Experiments/` の該当実験ノートを更新する。
- `ROGII_kaggle_obsidian/06_PDCA/Daily_Log/YYYY-MM-DD.md` にPlan / Do / Check / Actを記録する。
- 重要な判断は `ROGII_kaggle_obsidian/06_PDCA/Decision_Log.md` に記録する。
- Kaggle提出後は `ROGII_kaggle_obsidian/09_Submissions/LB_Tracking.md` を更新する。

## Obsidian記録ルール

- 「何を変えたか」「なぜ変えたか」「根拠は何か」「次に何をするか」を簡潔かつ具体的に書く。
- 失敗実験や捨てた案も記録する。失敗はプロジェクトの記憶である。
- LB確認用のリーク実験と、本物のvalidation結果を明確に分ける。
- `[[CV_Strategy]]`、`[[Anchor_Features]]`、`[[exp003_lgb_anchor_trajectory]]` のようなObsidianリンクを使う。
- Obsidianには巨大なraw data、parquet、model weight、OOF、submissionを置かない。必要ならリポジトリ内のパスで参照する。

## 公開リポ安全ルール

- commit前に `git status`、staged files、秘密情報スキャンを確認する。
- 公開リポに含めてよいのは、コード、設定テンプレート、軽量なドキュメント、fold定義など再現性に必要で安全なものだけとする。
- `.env.example` にはダミー値だけを書く。実値は絶対に書かない。
- GitHubなど外部リモートへのpushは、必ずユーザーの明示許可を得てから行う。
