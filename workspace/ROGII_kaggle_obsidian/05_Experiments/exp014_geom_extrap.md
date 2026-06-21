# exp014_geom_extrap (優先C / Group F)

## 概要
exp008 に **Group F = 幾何外挿特徴量** を追加。CV **13.808621 → 13.525189 (+0.283)**。
exp007以降で最大の単発改善。新best。

## 鍵となる気づき
**hidden(target)区間でも X/Y/Z/MD は完全既知（0% NaN）。隠れているのはTVTのみ。**
→ known区間で「TVTと幾何の関係（構造傾斜）」を較正し、既知hidden幾何へ投影すれば
   TVT変化量(delta)を幾何的に外挿できる。

## Group F 特徴量
per-well（known区間で較正）:
- `f_dtvt_dmd_l50` 直近50known行のTVT~MD傾き
- `f_dtvt_dz_pre` known全体のTVT~Z回帰傾き（全773well finite, mean -1.01, 構造傾斜の鉛直成分）
- `f_dtvt_dz_r2` 上記R²（外挿信頼度）

per-row（既知hidden幾何へ投影 → delta推定）:
- `f_extrap_slope20_dMD` = slope20 × delta_MD_from_PS
- `f_extrap_slope5_dMD`, `f_extrap_quad_dMD`(=slope·dMD+0.5·curv·dMD²)
- `f_extrap_z` = f_dtvt_dz_pre × delta_Z_from_PS
- `f_extrap_disagree` = |slope外挿 − Z外挿|（不確実性）

## 結果
| fold | rmse | best_iter |
|---|---:|---:|
| 0 | 12.675700 | 700 |
| 1 | 13.578982 | 158 |
| 2 | 13.711679 | 329 |
| 3 | 13.655823 | 629 |
| 4 | 13.969219 | 872 |

全fold一貫改善（fold0は13.04→12.68）。importance: f_dtvt_dz_pre #7, f_extrap_quad_dMD #8, f_dtvt_dz_r2 #9, f_extrap_slope5_dMD #11。

## なぜ効くか
delta = TVT − anchor が目的変数。`slope × delta_MD` 等の積特徴を明示的に与えると、
木が複数splitで再構成していた外挿量を直接表現でき、特に長尺wellで効く。
exp005の長尺hard well脆弱性に直接効いている可能性。

## リーク確認
構造傾斜の較正はknown行のみ。hiddenのX/Y/Z/MDは観測量、hiddenのTVTは不使用。
fold は exp008 と同一(folds_group_well_v001.csv)。leak-free。

## リンク
[[exp008_gr_rolling]] [[exp015_seq_smooth]] [[exp016_struct_plane]] [[Strategy_2026-05-31]]
