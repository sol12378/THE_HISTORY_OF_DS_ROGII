# 公開notebook監査 2026-06-10 (exp060後継)

5本のnotebookを `external_notebooks/` にpullし、並列worker監査を実施した。

## 系譜と構造

- **大元 = fleongg/fle3n-rogii-v4** (LB blend 7.572)
  - Engine A "Ridge-SP" (単独LB 7.776): ravaghi artifacts の学習済みLGBM×3+CatBoost×2 (GroupKFold OOF内蔵) → positive Ridge stack (α=1.660) 0.3 + selector lik-PF/beam 0.7 → U座標robust射影 (deg4, IRLS, β0.75)
  - Engine B "Drift-PF" (単独LB 7.810): ~185特徴 (PF_ancc/PF_z/beam×7/NCC/FormationPlaneKNN/DenseANCC/GR rolling/typewell template距離) のLGB平均 0.6 + likPF scale_5 0.4 + warmup(tau=85) + SG(61,3)。CV 9.21 (GroupKFold by well)
  - 最終: 0.55A + 0.45B
- **aiwody / jaemin3404 / pixiux の sp45-fleongg-blend-v2** = fle3n-v4の整理フォーク。jaemin=完全同一、pixiux=末尾にoverlap well上書きセル追加のみ。LB 7.4台
- **pilkwang/ridge-artifact-parameter-experiments** = 同資産のパラメータ実験ハーネス (プロファイル切替式、LB probe用候補CSV量産)

## 主要パラメータ (彼らの採用値)

- lik-PF: 128 seeds × 500粒子, scales (3,5,8,12), init_spread 4.5, MOM .998, VN .002, PN .005, 尤度softmax加重
- selector: n_eval>4840 / z_span (136.73, 185.51) の6bin → PF scale & hold(0.05-0.2)/beam重み切替
- warmup: delta *= 1-exp(-md_since/85)
- SG平滑: (17,3) ridge側 / (61,3) fleongg側
- projection: deg4, IRLS 4回 Cauchy重み, β=0.75 (我々のexp068/072はdeg5, β=1.0)
- Ridge meta: α=1.6602834637650032, positive=True

## リーク判定

| 項目 | 判定 |
|---|---|
| train TVT lookup (tvt_from_contacts) | **採点時は死にコード** (隠しtest wellはtrainに無い、exp044確定)。public LBは正味性能。移植禁止 (方針) |
| 特徴量・CV設計 | 健全 (GroupKFold by well、空間KNNはtrain時self除外、PF/beamはknown区間のみ) |
| blend重み 0.55 / selector閾値 / g_max | LB probing由来の疑い → **自前OOFで再決定** |
| pilkwang OOF (oof_imputer_mode) | self-well除外済みだがimputer自体はfold毎再構築せず → 軽微な楽観。許容範囲 (彼らのLB実績が裏付け) |

## 我々への取り込み (exp073)

1. ravaghi Trainer pickle の **OOF内蔵** → ridge stack OOFを正直に再構成可能
2. pilkwang package は **OOF一式同梱** (xgb/cat/hgb/lgb/TCN + blend, 3,783,989行) → 即ブレンド材料
3. Engine B はfold別再学習で正直OOF化 (likpf_mean_d 1特徴のみ欠落)
4. 我々の exp026 PF×geom は彼らに無い直交成分 → nested NNLSで統合
5. 後処理 (warmup/SG/射影β) はOOF ablationで再選択

## 関連

[[exp073_public_assets_integration]] [[Strategy_2026-06-10_LB7419_plan]] [[exp044]] [[exp068]]
