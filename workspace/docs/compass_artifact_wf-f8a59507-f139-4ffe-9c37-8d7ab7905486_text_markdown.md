# ROGII Wellbore Geology Prediction：地質・堆積学ドメイン理論と系列マッチング・アルゴリズム手法の深堀りレポート

## TL;DR
- **TVT予測の本質は「水平井GR系列を近傍typewellのGR-TVTプロファイルへ物理制約付きで貼り合わせる逐次トラッキング問題」であり、勝ち筋は機械学習回帰ではなく、dip・層厚・断層を状態変数とするベイズ的逐次推定（Particle Filter）+ DTW/Beam Search型グローバル整合 + 物理事前分布の三位一体である。** 公開LBは最良~8.86、Matteo Niccoli（LinkedIn）は "Leaders are sub-9 RMSE with particle filters and multi-model stacking." と明言。CV15のXGB/NNスターター（cdeotte）からの脱却にはこの転換が必須。
- **地質ドメインの効きどころは3つ**：(1) GR波形のmotif（funnel/bell/blocky/serrated）とparasequence stacking patternが「貼り合わせの拘束条件」になる、(2) maximum flooding surface（MFS）由来のhot shale高GRピークやmarker bedが「アンカー点」になる尤度設計、(3) signed azimuth（updip/downdip）とtortuosityが「系列の進行方向と探索幅」を決める—Niccoliは3D tortuosityが最有用ドメイン特徴と報告。
- **最大インパクトの打ち手は B（multi-typewell PF + dip高次化）と C（marker相関・GR尤度の物理的再設計）の統合**。PFは非線形・非ガウスの同時推定（層位変数+ツール位置）に最適で、"the estimation performance of sequential Monte Carlo estimator is not constrained by the nature of dynamics, measurement functions and the type of uncertainties"（Veettil & Clark 2020, Petrophysics 61(01):99–111）。打ち手D（参照解法のphysical model特定）はAlyaev MTP-loss CNN（hengck23が参照）とPluRaListic（RL+PF）が直接の設計図になる。

## Key Findings

1. **競技データは1ft刻みのTVT系列予測**。`horizontal_well.csv`はMD/X/Y/GR/TVT（target、訓練のみ）/TVT_input（評価ゾーンはNaN）、`typewell.csv`はTVT（垂直深度インデックス）/GR/Geology（例：EGFDL, BUDA）。各wellに`.png`断面図が付く。train **773井**（Niccoli, LinkedIn verbatim: "ROGII…with 773 horizontal wells with GR, XYZ trajectory, and a paired vertical type well"）、test約200井（Kaggle公式data page: "test/ Contains the evaluation data for about 200 wells."）、評価指標はTVTのRMSE（ft）。

2. **グローバルTVT-Z相関はr=−0.96だが井内ではほぼゼロ**（Niccoli）。これは「全体相関はビルドセクション幾何＝構造高度の井間差が支配し、井内の系列予測には直接効かない」ことを意味し、ナイーブな回帰が失敗する根本原因。well-levelの特徴量はGroupKFoldでcross-well noiseに過学習する。

3. **業界のTSP/TST法（Berg & Newson 2013, URTeC 1590259, doi:10.1190/urtec2013-121）が問題の物理定式化そのもの**：水平井のlog traceをtemplate（typewell）にsqueeze & invertして合わせる＝dip・層厚・displacementの調整。論文verbatim: "TST is superior to TSP because it can 'look ahead' of the bit … Vertical sections can have problems displaying apparent dip and horizons whereas displacement (curtain) sections will properly display both dips and horizons." ＝curtain（displacement）sectionがdipとhorizonを正しく表示する＝`.png`の読み筋。

4. **GR motifとsequence stratigraphyが拘束条件を与える**。funnel=上方粗粒化（progradation）、bell=上方細粒化（retrogradation/fining）、blocky/cylinder=均質砂、serrated=薄互層。MFSは「最高GRピーク」として現れるアンカー。Th/U比はredox（堆積環境）指標、hot dolomite（Permian）はU起因で200 API＝清浄砂岩の罠。

5. **アルゴリズムの最前線はPF・RL・MTP-loss CNN**。Veettil & Clark（SPWLA Petrophysics 2020, doi:10.30632/PJV61N1-2020a4）はSMC/PFで層位変数とツール位置を同時推定。PluRaListic（Muhammad, Cheraghi, Alyaev et al., SPE J. 30(03):995–1009, 2025, doi:10.2118/218444-PA）はRL+PFでROGII GWC人間専門家にベンチマーク（"PF continuously assimilates real-time log measurements…producing hundreds of most-likely geology interpretations"）。Alyaev & Elsheikh（Earth and Space Science 9(9):e2021EA002186, 2022, doi:10.1029/2021EA002186）のMTP-loss混合密度CNNは "trained using the 'multiple-trajectory-prediction' loss functions, which avoids mode collapse typical for traditional MDNs, and allows multi-modal prediction ahead of data"—hengck23が競技に応用中。

## Details

---
# パートI：地質・堆積学のドメイン理論

### I-1. GR log motif（funnel/bell/blocky/serrated）と堆積相
**中身**：GRは泥質（K-bearing illite、Th吸着、有機物固定U）で高く、清浄砂岩/炭酸塩で低い。Serra & Sulpice（1975）/Rider分類で、log shapeをsand body通過時のグレインサイズ・shaliness profileとして読む。3基本形＝funnel（coarsening-up, 上方GR減）、cylinder/blocky（均質）、bell（fining-up, 上方GR増）、これにsmooth/serrate（薄互層）の細分。Cant（1992）はbox/funnel/bell/bow/irregularの5分類。AAPG Wikiは「grain sizeはGRに直接効かず、shaliness（clay/mica）が効く」と明記—clay clast濃集や生物擾乱が誤読を生む。

**コンペへの応用**：horizontal GRをtypewell GRに貼り合わせる際、motifは「局所的な勾配の符号と形」を拘束する。funnelゾーンとbellゾーンは反転（squeeze/flip）の判定に直結し、wellがup-section/down-sectionどちらを通過しているかを規定する。

**効く打ち手**：C（GR尤度の物理的再設計）。点ごとのGR値一致だけでなく、局所微分（勾配符号）と窓内motif分類を尤度に加えることで、DTW/PFの誤対応（over-stretch）を抑制。B（PF）の観測モデルにmotif一致項を追加。

**想定インパクト**：中。motif項は平坦・単調区間での曖昧性を解消し、誤トラックによる大外れ（RMSEのtail）を削減。pooled RMSEはtailに敏感なので0.3-0.8ft改善が見込める。

### I-2. Sequence stratigraphy（parasequence, MFS, systems tract）
**中身**：parasequenceはflooding surfaceで境される最小単位。stacking patternがprogradational（funnel積み重ね）/retrogradational（bell）/aggradationalに分かれ、systems tract（LST/TST/HST）を構成。**MFSはretrograde→prograde転換面で、直上のcondensed shaleが最高GR応答を示す**（SLB glossary: "the shales that immediately overlie the maximum flooding surface commonly have different characteristics … recognized on the basis of resistivity, gamma ray, neutron and density logs"; KGS）。これらは時間面（chronostratigraphic）に近く、横方向に広く追跡できる＝well-to-well correlationのアンカー。

**コンペへの応用**：typewellのGRプロファイル上でMFS（最高GRピーク）や明瞭なflooding surfaceを「強アンカー」として抽出し、水平井で同じ高GRイベントを通過したらTVTを固定する事前情報にする。stacking patternの順序（prograde→MFS→retrograde）は系列の進行方向の制約。

**効く打ち手**：C（marker相関）。MFS/parasequence境界をpriorのknot（spline control point）にし、PFの粒子重み付けで「アンカー一致時に重みを集中」させる。

**想定インパクト**：中〜高。明瞭なmarkerがある井ではトラッキングのドリフト累積を断ち切れる。DTWの「path誤差累積・loop非閉合」問題（Sylvester 2023, Basin Research）への直接的対策。

### I-3. Spectral GR（K/Th/U）と岩相の罠
**中身**：全GRはshale指標として誤りうる。K-40（1.46MeV）、U系列（1.76MeV）、Th系列（2.62MeV）の3窓で分離。Th/K比でclay typing（kaolinite/illite/glauconite）、**Th/U比はredox指標**（Uは還元的条件で固定、酸化で可溶化—Adams & Weaver 1958, KGS）。罠：清浄だが放射性の砂岩（K長石・mica・zircon）、**hot dolomite（Permian basin、U起因で最大200 API＝shale様、AAPG Wiki）**、KCl泥水によるbaseline嵩上げ（約20 API）、steamflood時のradon性evanescent高GR。

**コンペへの応用**：本競技のtypewellはGRのみ（spectralは無い可能性大）だが、「全GRが岩相と一対一でない」という前提は尤度設計に効く。Eagle Ford（competのEGFDLラベル）はGRとTOCの相関が一貫せず（U・carbonate変動、SEG Wiki）、GRだけでlandingゾーンを切れない—よってmotif/相対パターン重視が正解。

**効く打ち手**：C（尤度のロバスト化）。絶対GR値マッチより、井内標準化（後述I-7）+相対パターン+ランク相関（Spearman/NCC）を尤度に。

**想定インパクト**：低〜中。直接特徴は無いが、絶対値依存を断つことで井間のGRスケール差による系統誤差を回避。

### I-4. 構造地質：dip・fault・foldとTVTの幾何
**中身**：TST = TVT·cos(dip)、apparent dip = ビット沿いに遭遇する見かけ傾斜。横方向に構造dipの変化・fold・fault offsetがあるとTVTが急変。Berg & Newson（2013）は「vertical sectionはapparent dip/horizon表示に問題、curtain/displacement sectionが正しい」と指摘。geosteering scaleの分解能はinchesオーダーで、seismicより桁違いに細かい（makinhole.com）。fault通過は「予測logと実測の大不一致」として現れ、modelにoffsetとして組み込む（Drilling Contractor）。

**コンペへの応用**：TVTの時間微分（MDに対する変化率）＝局所apparent dipは状態変数。dipを定数でなく区分的（piecewise）/高次（線形・スプライン）にモデル化することで、構造変化に追従。fault offsetは「TVTの不連続ジャンプ」として許容（ペナルティ付き）。

**効く打ち手**：B（dip高次化）。PF/DPの状態を(TVT, dip, dip変化率)に拡張し、不連続スプライン（Maus/Gee系のheatmap法、H&P/Willerth）でfault offsetを表現。Niccoliの「signed azimuth matters（updip/downdipで層を逆順に見る）」を方向符号として状態に組込む。

**想定インパクト**：高。dipの硬直モデルは長いlateralで誤差が線形累積する主因。高次化はLBの大きな改善源。

### I-5. Cyclostratigraphy / 波形の周期性
**中身**：GR系列にはMilankovitchサイクル（405kyr長偏心、~100kyr短偏心、obliquity、precession）が記録されることがあり、CWT・FFT・spectral analysisで検出（Sichuan/Bohai Bay事例）。ただしfalse cyclicity検出のType I error問題がある（Sci. Direct 2022）。

**コンペへの応用**：周期性は「typewell GRの自己相似的な繰り返しパターン」が誤対応（aliasing）を生む危険を示唆。DTW/相関が周期パターンで複数の極小に陥る＝multi-modal。これはむしろ「単峰探索の危険」を裏付け、multi-hypothesis（Beam/PF）を正当化する。

**効く打ち手**：A/B。CWT/DWT特徴（既に公開notebook「9.251 DWT-based」が使用）でスケール分離した相関を取る。multi-hypothesis tracking（Beam search）でaliasing極小を同時保持。

**想定インパクト**：低〜中。DWT特徴は公開LB ~9.25で確認済み。aliasing対策としてのmulti-hypothesisは間接的だが堅牢化に寄与。

### I-6. Chemostratigraphy / marker bed同定
**中身**：XRF元素比（Zr/Y, Th/Nb, Si/Al, Ca等）で、marker bedが乏しい厚いshale層でも相関可能（Bruker, Rowe et al. 2012）。U濃度は有機物量と相関し相関アンカーになる。Wolfcamp（competのターゲット級プレイ）はcoarsening/fining cycleとbasin-wide shale marker（MFS）で枠組み化（Search & Discovery, Delaware Basin）。

**コンペへの応用**：本競技にXRFは無いが、「GR上で識別可能なmarker bed（薄い高/低GRスパイク）を自動抽出してアンカー化」という発想を借用。Geologyラベル（typewellに有り）が層境界の地上真実を提供。

**効く打ち手**：C（marker相関）。typewellのGeologyラベル境界＝formation topをアンカーとし、水平井で対応するGRイベントにTVTを固定する制約付き整合。

**想定インパクト**：中。Geologyラベルは訓練の強い教師信号。境界アンカーはトラッキング初期化と再同期に有効。

### I-7. GR正規化・baseline drift補正
**中身**：井間でGRのAPIスケール・baselineが系統的にずれる（ツール較正、泥水、孔径）。well-to-well correlationの前処理として、histogram正規化・min-max・分位点マッチング・detrend（多項式/CWT detrend、Bahmaei 2019のDTEL/INDTEL）が標準。

**コンペへの応用**：horizontal GRとtypewell GRのスケール整合は貼り合わせの前提。絶対値マッチを使うなら必須。

**効く打ち手**：C（前処理）。井ペアごとの分位点正規化＋robust z-score。公開notebookに「Drift Targeting + NCC」（LB 8.905）が存在＝drift補正+正規化相互相関が有効と実証済み。

**想定インパクト**：中。前処理だけでLB 8.9台が出ている事実は、正規化が高インパクトであることの直接証拠。

---
# パートII：波形相関・系列マッチングのアルゴリズム手法

### II-1. Dynamic Time Warping（DTW）とその制約版
**中身**：DTWはquery（歪んだ系列）をreferenceにstretch/compressして整合する動的計画法（Sakoe & Chiba 1978）。over-stretch/compressの病理を防ぐため、グローバル制約（Sakoe-Chiba band＝対角帯、Itakura parallelogram）、ローカル制約（step pattern/slope weight）、特徴空間変換（shapeDTW, derivative DTW, ESDTW=extrema-based shape DTW）がある。Sakoe-Chiba bandは一般にItakuraより良好（Geler et al. 2019）。Sylvester（2023, Basin Research）はDTW pairwiseをRGT（relative geologic time）最小二乗（conjugate gradient）に統合してmulti-well整合し、「path誤差累積・loop非閉合」問題を解決。GR-CWT detrend + DTWの実装例（Bahmaei 2019, J. Appl. Geophysics）も。

**コンペへの応用**：horizontal GR↔typewell GRの整合の核。warping pathの傾き＝局所のstratigraphic rate（dip/squeeze比）。band幅＝許容dip範囲の事前。

**効く打ち手**：B/C。constrained DTW（band幅をdip物理から設定）+ derivative/shape DTWでmotif一致を強化。単独DTW（=DPトラッキング）はNiccoliが「leadersにgapがある」と報告＝DTWだけでは不十分、PFとの併用が必要。

**想定インパクト**：中。DTWは強いベースラインだが、グローバル最適単独では局所多峰性に弱い。

### II-2. Particle Filter / Sequential Monte Carlo（逐次ベイズ）
**中身**：PFは状態（層位変数+ツール位置）の事後分布を粒子集合で近似する逐次推定。非線形・非ガウスでもKalmanの制約を受けない（Veettil & Clark 2020, Petrophysics 61(01):99–111, doi:10.30632/PJV61N1-2020a4: "the estimation performance of sequential Monte Carlo estimator is not constrained by the nature of dynamics, measurement functions and the type of uncertainties"）。観測（GR）を逐次同化し、粒子をstratigraphy変化モデル（heuristic prior）で前進。粒子数が精度を決める。geosteering inversionの主流定式化（Alyaev 2022がレビュー：Winkler Bayesian network, Gee/Maus heatmap, Veettil PF）。

**コンペへの応用**：MDに沿ってTVTを逐次推定する自然な枠組み。状態=(TVT, dip, dip変化率)、観測モデル=「現在TVTでtypewell GRを引いた予測GR vs 実測GRの尤度」、遷移モデル=dip連続性+fault offset事前。multi-modal（複数解）を保持できるのがDTW単独に対する優位。

**効く打ち手**：B（PF本体）。公開notebook「Inference Stack with PF, Beam and TabICL」（LB 9.349）、「Sunny Physical PF」「Particle Filter Parameter Study」（~9.15）が実在＝PFが上位アプローチと実証済み。リーダーは "particle filters and multi-model stacking"（Niccoli）。

**想定インパクト**：最高。CV15回帰→sub-9へのジャンプの主因。PF+物理事前が現状の勝ち筋の中核。

### II-3. RL+PF統合（PluRaListic）とMTP-loss CNN
**中身**：(a) PluRaListic（Muhammad, Cheraghi, Alyaev, Srivastava & Bratvold, SPE J. 30(03):995–1009, 2025, doi:10.2118/218444-PA；RL+PF版はComput. Geosci. 29:14, 2025, doi:10.1007/s10596-025-10352-y / arXiv 2402.06377）：PFがリアルタイムlogを同化し "hundreds of most-likely geology interpretations" を生成、RLが操舵を最適化。ROGII GWC 2021の人間専門家ベンチで上位四分位超え。(b) Alyaev & Elsheikh（Earth and Space Science 9(9):e2021EA002186, 2022, doi:10.1029/2021EA002186）：混合密度DNN（MDN）を**multiple-trajectory-prediction（MTP）loss**で訓練し "avoids mode collapse typical for traditional MDNs, and allows multi-modal prediction ahead of data"、GR logのmulti-modal stratigraphic inversionを単一forward passで実現。hengck23が競技スレで「MTP with deep CNN for welllog inversion」として参照。

**コンペへの応用**：本競技は操舵（RL）不要のオフライン予測だが、(a)のPF部分と(b)のmulti-modal予測は直接転用可。MTP-loss CNNは「複数の妥当なTVT軌道+各確率」を出力し、PFの初期化/提案分布や尤度の代理に使える。GWC 2021データで訓練した確率的地質モデル（PluRaListicが使用）は事前分布の実証ソース。

**効く打ち手**：D（参照解法の物理モデル特定）＋A（CNN）。MTP-loss CNNでmulti-modal事前を作り、PFに供給。

**想定インパクト**：高。これが「参照解法のphysical model」の正体に最も近い。多峰予測はaliasing/dip反転の曖昧性に直接効く。

### II-4. HMM / Viterbi / 動的計画によるセグメンテーション
**中身**：HMMで隠れ状態（formation/stratigraphic position）系列を、Viterbiで最尤経路、forward-backwardで周辺確率を推定。k-segment制約やP-best list（複数解）拡張がある（Titsias et al. 2016, JASA）。Bayesian版はMCMC/Gibbsで自由度を積分。

**コンペへの応用**：TVTを離散化（または区分線形）し、状態遷移＝dip連続性、emission＝GR尤度とすればViterbiでグローバル最尤TVT経路。Beam searchはViterbiの幅制限近似でmulti-hypothesisを保持。

**効く打ち手**：B。Beam search（公開「PF, Beam and TabICL」が使用）はViterbi/DPのmulti-hypothesis版で、PFと相補的。

**想定インパクト**：中〜高。DP/Viterbiはグローバル最適、PFは逐次・非ガウス—両者stackingが堅牢。

### II-5. グローバル最適 vs 逐次推定の比較
**中身**：DTW/Viterbi=系列全体のグローバル最適（loop非閉合・誤差累積に注意、Sylvester 2023）。PF/SMC=逐次・オンライン、非ガウス可、multi-modal保持、粒子数依存。Kalman/EKF/UKFはガウス・弱非線形限定。

**コンペへの応用**：評価はオフラインなので双方使える。グローバル法は「typewell全長との整合」に、逐次法は「dip漸変・fault・aliasing」に強い。両者の予測をstackingするのが定石（リーダーのmulti-model stacking）。

**効く打ち手**：全打ち手の統合層。DTW（粗整合）→PF（精緻化）→DP/Beam（グローバル検証）→stacking。

**想定インパクト**：高。単一手法の弱点を相互補完。

### II-6. Formation top自動検出の深層学習
**中身**：1D-CNN（GeoConvention 2025、ensembleで信頼度+精度向上）、constrained CNN（SEG 2019、prior分布でpick拘束）、Soft-Attention CNN（SPE ATCE 2021）、CNN+BiGRU+self-attention（欠損log補完）、CNN+LSTM時空間NN（STNN, J. Geophys. Eng. 2021）。教師ペアデータ作成が高コストで、unsupervised法（近傍井の類似性利用、SPE ADIP 2020）も。

**コンペへの応用**：typewellのGeologyラベル境界（formation top）をCNNで水平井GRから検出し、アンカー/再同期点に。multi-well joint correlationの発想（複数typewell）も。

**効く打ち手**：A（CNN）+C（marker）。ただしNiccoliは「per-formation GR classifierはほぼ無寄与」と報告＝単純分類は効かず、トラッキングの補助に留めるべき。

**想定インパクト**：低〜中。単独では弱いが、アンカー供給として価値。

### II-7. PNG断面図のCNN regression（打ち手A）
**中身**：各wellの`.png`はcurtain section（trajectory+horizon+target zone+解釈点）を描画。画像CNN（U-Net系horizon auto-tracking、地震解釈のhorizon追跡技術）で構造情報を抽出可能。

**コンペへの応用**：`.png`にはTVT解釈そのものに近い幾何（dip, horizon, well path）が描かれる可能性が高く、CNN regressionで構造priorを抽出。ただしtest側でリークの無い範囲（評価ゾーンが描かれているか）の確認が必須。

**効く打ち手**：A。画像から構造dip・horizonの大局形状を回帰し、PFのdip事前に供給。

**想定インパクト**：中（不確実）。`.png`が評価ゾーンの解答を含むならリーク、含まないなら構造priorとして中程度。要検証。

### II-8. 信号処理：特徴点抽出・正規化・アンサンブル
**中身**：peak/勾配反転（極値）抽出（ESDTW=extrema-based shape DTW）、baseline detrend（CWT polynomial、DTEL/INDTEL、Bahmaei 2019）、正規化相互相関（NCC）、DWT多重解像度特徴。アンサンブル/stackingで系列予測を堅牢化。

**コンペへの応用**：極値列のマッチは高速かつdrift頑健。NCC+drift補正は公開LB 8.905で実証。

**効く打ち手**：C（前処理・特徴）+全体stacking。

**想定インパクト**：中。前処理品質がトラッキング精度の上限を決める。

---
## 競技固有の重要事実（subagent確認）
- **train 773井 / test 約200井**、TVTは1ft刻み、指標はTVTのRMSE（ft）。Geologyラベル例：EGFDL（Eagle Ford系）, BUDA。
- **公開ベースライン**：cdeotte XGB Starter CV~15、NN Starter CV~15.5。公開最良notebook ~8.86（romantamrazov: SUPER BASELINE 12.602→BETTER 9.956→SUPER SOLUTION TOP 3）。
- **上位の構成要素**：PF、Beam search、DTW/DP、DWT特徴、NCC、tree（XGB/LGBM/CatBoost）、TabICL、physical model、multi-model stacking。
- **Niccoliの実証的教訓**：global TVT-Z r=−0.96だが井内ゼロ；3D tortuosityが最有用ドメイン特徴；per-formation GR classifierはほぼ無寄与；signed azimuth（updip/downdip）が重要；CVはStratifiedGroupKFold（signed azimuth・median TVT・空間位置で層化）が適切（validation井がtrainと空間的に交互配置＝補間条件のため、空間ブロックは過度に悲観的）。

## Recommendations

**ステージ1（即時、最大ROI）— PF + 物理事前の核を立てる（打ち手B+C）**
1. 状態 = (TVT, apparent dip, dip変化率)。観測モデル = typewell GRを現TVTで引いた予測GR vs 実測GRの尤度（井ペア分位点正規化後）。遷移モデル = dip連続性ガウス + fault offsetの重尾混合。
2. 尤度に「局所勾配符号」「窓内motif（funnel/bell/blocky）一致」を加える（I-1, I-4）。
3. typewellのGeologyラベル境界をアンカー化し、対応GRイベントで粒子重みを集中（I-2, I-6）。
4. signed azimuthを状態に組込み、updip/downdipで探索方向を切替（Niccoli）。
- **ベンチマーク**：このPFがCV ~9を割れば方向性正、割れなければ尤度/遷移モデルを再設計。

**ステージ2 — グローバル整合とmulti-hypothesis（打ち手B）**
5. constrained DTW（Sakoe-Chiba band=dip物理幅）で粗整合→PF初期化。derivative/shape DTWでmotif強化。
6. Beam search（Viterbi幅制限）でaliasing/dip反転の複数仮説を保持し、PF解と相互検証。
- **ベンチマーク**：DTW単独でleadersにgap（Niccoli）なら必ずPFと統合。

**ステージ3 — multi-modal CNNと参照解法特定（打ち手A+D）**
7. Alyaev MTP-loss混合密度CNN（doi:10.1029/2021EA002186）を実装し、multi-modal TVT軌道+確率を出力→PFの提案分布/事前に供給（hengck23路線）。
8. `.png` curtain sectionをCNN regressionで構造dip prior抽出—**ただし評価ゾーンのリーク有無を最優先で検証**。
- **ベンチマーク**：CNN priorでPFのtail誤差が減るか。減らなければCNNはアンカー供給に限定。

**ステージ4 — stacking堅牢化**
9. PF / DP-Beam / DTW / tree（drift+NCC特徴）/ TabICL の予測をStratifiedGroupKFold（signed azimuth・median TVT・空間位置で層化）でstacking。
10. tail（大外れ）に効くアンカー再同期とfault許容を調整—pooled RMSEはtail支配のため。
- **閾値**：CV換算~7.9（LB 6.5級）に届かなければ、dip高次化（スプライン）とmarkerアンカーの密度を上げる。

## Caveats
- **公開情報の上限**：公開notebookは~8.86止まり、Niccoliは "Leaders are sub-9 RMSE with particle filters and multi-model stacking" と表現。タスク前提の「1位LB 6.534」は公開ソースで直接確認できず、private/最新LBの値として扱うべき（subagentも未確認とフラグ）。
- **`.png`のリークリスク**：断面図が評価ゾーンのTVT解釈を描いている場合、CNN利用はリークになりうる。test側の`.png`内容を必ず確認。
- **データ列の未確認部分**：horizontal_well.csvは「trajectory, geological surfaces」を含むと公式記載だがTVD/inclination/azimuthや具体的なsurface列名はverbatim未取得。実データで要確認。
- **指標のpooled vs macro**：id={well}_{row}の単一LBスコアからpooled RMSEと推定されるが、公式文言でpooled/macroは明記未確認。
- **地質ドメインの転用限界**：spectral GR・XRF・cyclostratigraphyの直接データは競技に無い可能性が高く、これらは「尤度設計の発想」「絶対GR依存を断つ根拠」として間接利用するもの。Eagle Ford等でGR-TOC相関が一貫しない事実（SEG Wiki）は絶対値マッチの危険を裏付ける。
- **per-formation classifier無効**：Niccoliの実証通り、地質分類を直接特徴にしても効かない。地質知識はあくまで「トラッキングの拘束・尤度・アンカー」として組込むのが正しい使い方。