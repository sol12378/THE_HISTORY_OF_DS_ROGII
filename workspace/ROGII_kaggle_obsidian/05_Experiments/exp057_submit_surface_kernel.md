# exp057_submit_surface_kernel

> [!warning] CANCELLED (2026-06-10)
> 前提の [[exp056_field_surface]] CV 0.626 は **self-leak** (cKDTree構築時に評価well自身のASTNU点が残存、正しいLOWO=pooled 29.2でPF以下) と2026-06-07のDecision_Logで確定済みだったため、本実験は実装前に中止。後継戦略は [[exp073_public_assets_integration]]。

## 仮説 (無効)

[[exp056_field_surface]] のformation horizon surface法をKaggle kernelへ移植すれば、既存提出best LB 8.672を大きく下回り、LB 7.419以下を狙える。

## 目的

ASTNU surface単体を第一候補として、test wellのknown TVT区間から `const_well` を推定し、hidden TVTを予測する提出用pipelineを作る。

## 変更予定

- `experiments/exp057_submit_surface_kernel/` を作成する。
- exp056のLOWO検証を再現する。
- Kaggle notebook用の自己完結コードへ整理する。
- `result.json`, `oof.csv`, `submission.csv`, `notes.md` を保存する。

## 検証方針

- LOWOで評価対象wellをsurface構築から除外する。
- hidden TVTを `const_well` 推定に使わない。
- sample submission形式、row数、NaN、重複idを確認する。

## リーク懸念

低。ただし、formation horizonの扱いがtestで許容される入力データに限定されているかを提出前に再確認する。公開notebookの外部予測値を混ぜる場合は別実験に分離する。

## 採用条件

- LOWO pooled CV <= 0.8 ft。
- Kaggle kernel制限内で実行可能。
- exp026 LB 8.672より明確に改善する見込みがある。
