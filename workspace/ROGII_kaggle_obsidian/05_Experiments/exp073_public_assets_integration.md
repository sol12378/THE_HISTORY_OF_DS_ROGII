# exp073_public_assets_integration

## 仮説

公開資産 (ravaghi artifacts / fleongg models) から fleongg blend (LB 7.572) の主成分の**正直なOOF**を構築し、我々の直交成分 exp026 (PF×geom, 単独LB 8.672) と nested NNLS blend すれば、現best CV 9.086 (exp072) を更新し、LB 7.419 以下に到達できる。

根拠:

- fleongg Engine B 単体 CV 9.21 (GroupKFold by well) ≈ 我々の exp051 blend (9.27) と同等の独立した強成分。
- 我々の exp026 PF×geom は彼らの blend に含まれない (彼らの PF は lik-PF 系で実装・パラメータが異なる)。
- fleongg 自身が「相関 ρ≈0.89 の2エンジンでも blend で -0.2 ft」と報告。直交性のより高い成分追加で同等以上の改善が見込める。

## 構成ステップ

- exp073a: 我々のOOF再構築 (geom LGB 5-fold + PF 128seed×500粒子 全773well)。Windowsマシンに過去OOFが無いため。**PFは重い (~1h並列) — 事前明記**
- exp073b: ravaghi Trainer pickle から oof_preds 抽出 + ridge-stack OOF再構成 (positive Ridge α=1.66)
- exp073c: Engine B 相当のLGBをravaghi train.csvの特徴量でfold別再学習 → 正直OOF (公開モデルはall-train学習のためOOF不可)
- exp073d: 後処理ablation (warmup tau=85 / SG平滑 / projection deg・β) を OOF上で1変更ずつ
- exp073e: nested NNLS blend。**CV < 9.086 でのみ提出へ**
- exp074: 提出kernel (exp072 wrapper のblendセクションに注入 or sp45-fleongg-v2 fork)

## リーク懸念

- ravaghi train.csv 特徴量のself除外は検証不能 (見える範囲のコードは leave-self-out 実装済み)。OOF blend重みの異常な偏りで検知する。
- Engine B 再学習は well単位 GroupKFold 厳守。typewell重複well ([[exp011_typewell_leak_test]]) に注意。
- train lookup / overlap override 系コードは移植しない (採点時死にコードだが方針として排除)。
- worker成果物のleak自己検証必須 (exp056/self-calibの教訓: [[Strategy_2026-06-10_LB7419_plan]])。

## 採用条件

- 各成分OOFのfold/well整合検証パス。
- nested blend CV < 9.086 (現best) かつ fold一貫。
- 提出kernelのローカル予測と submission 形式検査パス。

## 関連

[[Strategy_2026-06-10_LB7419_plan]] [[exp026_final_blend]] [[exp051]] [[exp068]] [[exp071]]
