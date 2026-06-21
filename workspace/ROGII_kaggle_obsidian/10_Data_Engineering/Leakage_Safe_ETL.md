# Leakage-Safe ETL

## リークの定義

test prediction時点で利用できない情報をfeatureに使うことをリークとする。

## 高リスクな情報源

- future `TVT`
- post-PS `TVT_input`
- train-only formation marker columnsを通常特徴量として使うこと
- train-only `Geology`
- trainに重複しているpublic test wells
- row-random validation

## Safe Anchor Construction

各wellについて:

1. `ps_idx` を検出する。
2. anchorとpre-PS slopeには rows `< ps_idx` だけを使う。
3. rows `>= ps_idx` からはtest-timeで使える現在行の `MD`, `X`, `Y`, `Z`, `GR` だけを使う。

## Safe Rolling Features

GR rolling featuresは、testでも存在するGR値に基づく限り利用できる。

TVT rolling featuresはpre-PSの観測済み `TVT_input` に限定する。hidden train `TVT` を使ったtarget-side rolling statisticsは、analysis-onlyとして明記しない限り使わない。

## Validation Safety

target-aware aggregationを作る前にgroup splitを行う。wellをまたいでtarget情報を集約するfeatureはfold-wiseに計算する。
