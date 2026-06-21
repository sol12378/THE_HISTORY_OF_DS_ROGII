# exp015_seq_smooth (優先B / 系列平滑化)

## 概要
予測TVTを well内 row_idx 順に平滑化する後処理。**完全leak-free**（既存予測の平滑化のみ）。
手法(median/mean/savgol)×窓(5〜101)をOOFでsweep、fold一貫性も確認。

## 動機
真のTVTは MD に沿って極めて滑らか（隣接 |ΔTVT| 中央値0.01、75%点0.02）。
一方LightGBM予測は隣接差 平均0.029・max20 とジッター/ジャンプを含む。
→ 平滑化で真の滑らかさに寄せRMSE改善。

## 結果

### on exp008 (CV 13.808621)
best = mean w=101 → **13.803155 (+0.0055)**。全5fold一貫改善(+0.004〜+0.007)。

### on exp014 (CV 13.525189, 新best)
best = mean w=101 → **13.520383 (+0.0048)**。全5fold一貫改善(+0.0045〜+0.0054)。
test 14151行に適用。

## 判定
小さいが**頑健**な改善（exp012 anchor guardと違いfold全勝・過学習なし）。
mean w=101 という単一パラメータなので過学習リスク極小。最終blendに安全に組込み可。
**現best = exp014 + exp015平滑化 = 13.520383**。

## 注意
窓101は広い。真のTVTジャンプ(まれ)を鈍らせる懸念はあるが、OOF全体・fold別で
一貫改善のため正味プラス。将来モデルが変わったら window を再sweepする。

## リンク
[[exp014_geom_extrap]] [[exp012_anchor_guard_exp008]] [[Strategy_2026-05-31]]
