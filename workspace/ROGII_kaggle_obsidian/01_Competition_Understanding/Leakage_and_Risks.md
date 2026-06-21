# リークとリスク

## 既知のリークリスク

現在ダウンロードできるtest wellsはtrainにも存在する。

- `000d7d20`
- `00bbac68`
- `00e12e8b`

train filesにはsample submission対象行の `TVT` が含まれる。直接lookupするsubmissionは必ず `leak_risk` と明示し、提出形式確認用としてのみ扱う。

## Train-only columns

以下のhorizontal well columnsはtrain-onlyである。

- `ANCC`
- `ASTNU`
- `ASTNL`
- `EGFDU`
- `EGFDL`
- `BUDA`
- `TVT`

これらは通常の本番特徴量としては使わない。分析や補助目的で使う場合も、リーク懸念を明記する。
