# Strategy 2026-06-10: LB 7.419 以下を狙う実験計画 (改訂版)

## 重要な訂正 (当初計画の棄却)

当初この文書は [[exp056_field_surface]] (CV 0.626) の提出kernel化を主戦略としていたが、**exp056の0.626はself-leakであることが2026-06-07のDecision_Logで既に確定していた**ため、当初計画を全面棄却する。

- leakの実体: cKDTree構築時に評価対象well自身のASTNU点が除外されていない (`scripts/exp056_field_surface_full.py` にself除外コードが存在しない)。formation horizon値は自wellのTVT解釈由来のため、自己参照でTVTを当てているに等しい。
- 正しいLOWO (self除外) = pooled 29.2 / median 12.3 ft で、PF (11.024) に明確に劣る。
- 教訓: 戦略立案時はEXP_SUMMARYだけでなくDecision_Logの棄却履歴を必ず突合する。

## 検証済みの現状 (2026-06-10)

- **現best LB = exp072 8.280** (ref 53496203, exp051 blend + projection)。
- 現best CV = nested 9.086 (leak-free)。CV-LB gap ≈ 0.81。
- 目標 LB <= 7.419 まで **残り -0.861**。
- 過去のCV→LB転移は安定 (gap 0.8〜1.5、CV改善はほぼ1:1でLBに転化)。

## 公開notebook監査の結論 (5本、external_notebooks/ にpull済み)

| notebook | 実体 | LB |
|---|---|---|
| fleongg/fle3n-rogii-v4 | 大元。Engine A (ravaghi ridge-stack 0.3 + selector lik-PF 0.7 + projection) × Engine B (150特徴LGB 0.6 + likPF scale5 0.4) を 0.55/0.45 blend | A単独 7.776 / B単独 7.810 / blend **7.572** |
| aiwody / jaemin3404 / pixiux の rogii-sp45-fleongg-blend-v2 | fle3n-v4の整理版フォーク (jaemin=同一、pixiux=overlap上書き追加のみ) | 7.4台 (=目標値の出どころ) |
| pilkwang/rogii-ridge-artifact-parameter-experiments | 同資産のパラメータ実験ハーネス (σ0/deg/λ/g_maxのLB probing用) | 主張なし |

リーク監査:

- **train TVT lookup分岐 (tvt_from_contacts) は採点時には発火しない死にコード**。採点対象の隠しtest wellはtrainに存在しない (exp044で確定)。よって彼らのpublic LBは正味のアルゴリズム性能であり、信頼してよい。
- ただし blend重み (0.55等)・selector閾値 (n_eval 4840 / z_span 136.73, 185.51)・g_max はLB probingで選ばれた可能性が高く、**重みは自前OOFで決め直す**。
- 彼らの特徴量・CV設計自体は健全 (GroupKFold by well、空間KNNはtrain時self除外、PF/beamはknown区間のみ使用)。

## 新戦略: exp073 公開資産統合 (public assets integration)

鍵となる事実: 彼らのpipelineは全て公開Kaggleデータセットで構成されている。

- `ravaghi/wellbore-geology-prediction-artifacts`: train特徴量テーブル (7.4GB) + LGBM×3/CatBoost×2 のkoolbox Trainer pickle (**GroupKFold OOF予測を内蔵** → 正直なOOFが即入手可能)
- `fleongg/rogii-claude-models-pub`: Engine B の学習済みLGB×3 + features.json
- `pilkwang/rogii-model-package`: fold別モデル + 特徴量生成コード (rogii_feature_core.py)

我々の優位性: **exp026 (PF×geom, 単独LB 8.672) は彼らのblendに入っていない直交成分**。我々のprojection (exp068/072) は彼らと同系だが、selector (exp071, PF 10.91→10.47) も独立実装済み。

### 実験ステップ

1. **exp073a 我々のOOF再構築** (このマシンにOOFが無いため): exp026のgeom LGB 5-fold OOFとPF (128seed×500粒子, 全773 well) を再計算。PFは重い (~1h, 並列) — 事前明記。
2. **exp073b 彼らのOOF抽出**: ravaghi Trainer pickleから oof_preds を抽出し、train.csv の行と整合検証。ridge-stack OOFを再構成 (positive Ridge α=1.66)。
3. **exp073c Engine B OOF**: ravaghi train.csv の特徴量 (features.json で選択) で fleongg Engine B 相当のLGBをfold別に再学習し正直OOFを得る (彼らの公開モデルはall-train学習のためOOFに使えない)。
4. **exp073d 後処理ablation**: warmup damping (tau=85)、SG平滑 (17/61)、projection (deg4 β0.75 vs 我々のdeg5 β1.0) をOOF上で1変更ずつ検証。fold一貫性ゲート必須。
5. **exp073e nested NNLS blend**: {ridge-stack, EngineB, exp026 PF, geom, (likpf selector)} のnested-fold blend。**CV < 9.086 を更新した場合のみ提出へ**。
6. **exp074 提出kernel**: exp072 wrapperの注入点 (blendセクション) に新成分を追加、または sp45-fleongg-v2 をforkして我々の成分を第3エンジンとして注入。重みは自前CVで決定。

### 採用条件 / リスク管理

- 各成分OOFはGroupKFold by well / LOWOで構築し、自己参照leakを成分ごとに監査する (worker成果物は必ずmainがleak検証)。
- blend重みのfold安定性ゲート (pdca_blend基盤を再利用)。
- 期待値: fleongg blend LB 7.572 ≈ CV ~8.9相当。これに直交成分exp026 (corr ~0.6-0.85想定) を正直な重みで加えれば CV 8.5-8.8 → LB 7.2-7.5 が現実的レンジ。
- 彼らのLB probing値の鵜呑み禁止。CVで決めた重みのみ採用。

## GPU解禁 (2026-06-10追記)

CUDA GPU (RTX 2080 SUPER 8GB) が利用可能に。XGBoost CUDA / CatBoost GPU 動作確認済み、LightGBMはpip wheel非対応 (CPU)、torch CUDA導入済み。

GPUで解禁される実験 (LB 7.419への寄与順):

- **exp075 GPU系列ニューラル (本命)**: torch CUDAで well の MD系列 (GR/幾何/typewell prior) → TVT delta を学習 (geom prior残差)。pilkwang勝ちblendが sequence_tcn を含む=TCN系統はblend+後処理で有効と実証済み。過去のexp019/020/039/046はMac/MPS制約で頓挫したが、CUDA実機で再挑戦。**「非相関な第3ソース」(著者が改善の唯一の道と明言) の最有力候補**。
  - 位置づけ: exp073 blend harness (pilkwang TCN OOF を既に含む) が 9.086 を僅差でしか超えない場合の上乗せ。comfortably超える場合は優先度を下げる。
  - leak管理: well単位GroupKFoldで正直OOF。geom priorへのresidual学習。
- exp073のengineB等のGBM再学習はXGB/CatBoost GPUで高速化可能 (ただしOOF抽出済みのため優先度低)。
- 採点kernelもKaggle GPU(T4)利用可 → 外部pipelineのGPU推論を実機で再現検証できる。

## リーク懸念

- ravaghi train.csv の特徴量が本当にself除外で作られているかは検証不能 (low risk評価だが、OOF blend重みが異常に偏る場合は疑う)。
- Engine B再学習時のfold分割はwell単位GroupKFoldを厳守。typewell重複well (exp011) のリークに注意。
- train lookup系・overlap override系のコードは一切移植しない。

## 次アクション

1. 外部データセット4つのダウンロード完了を確認。
2. exp073a/b/c をworkerに委譲し並列実行。
3. exp073e でCV更新を確認したら、ユーザー事前許可に基づき提出。
