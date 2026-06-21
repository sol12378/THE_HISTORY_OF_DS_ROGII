# ROGII Wellbore Geology Prediction — geosteering ドメイン知識レポート

> 目的: このコンペの本質が **geosteering（ジオステアリング）の自動化** であることを踏まえ、
> 1位（LB 6.534 / CV ≈ 7.9）に到達するために必要なドメイン知識を体系的に整理する。
> 各セクションは、別途まとめた打ち手 **A（PNG断面図）/ B（multi-typewell + dip）/ C（marker相関・尤度再設計）/ D（参照解読）** に
> 直接ひもづけて書いている。「知識 → なぜCVに効くか」をペアで読めるようにした。

---

## 0. 結論（最初に読む3行）

1. このタスクは汎用MLではなく、石油業界で確立した **TVT空間でのGR波形相関（log correlation）** を解く問題。参照解法が「機械学習ではない系列トラッカー」なのは必然。
2. CVの壁（per-well offset がleak-free信号で予測不能、47壊れwell）は、ドメイン的には **「typewellの割り当てとdip推定が間違っている」** ことが主原因の可能性が高い。これは打ち手B（multi-typewell PF）で直接攻められる。
3. ROGII自身のソフト（StarSteer / GeoAssist）が「**自動で最も近いtypewellを選び、複数offset wellから複数解釈を作り最良相関を選ぶ**」設計。これがそのまま、あなたが未実装の方向を示している。

---

## 1. 基礎物理 — GR（ガンマ線検層）が見ているもの

### 1.1 何を測っているか
GR = Gamma Ray log。地層が自然放出するガンマ線量を坑井に沿って測る。単位は API（gAPI）。
線源は岩石中の放射性元素 = **カリウム(K)・トリウム(Th)・ウラン(U)**。標準GRの定義式はおおむね：

```
GR (API) ≈ 8 × U[ppm] + 4 × Th[ppm] + 16 × K[wt%]
```

粘土鉱物がこれらを吸着・保持するため、**泥質ほどGRが高い**。

### 1.2 岩相とGRの対応（基本則）
- **GR高い → シェール・粘土（泥質）**
- **GR低い → 砂岩・石灰岩・ドロマイト（清浄な砕屑岩/炭酸塩）**

この対比により **GRカーブの形が地層の「指紋」** になり、隣接井どうしの **well-to-well correlation（検層対比）** が成り立つ。これが石油地質学で100年近く使われてきた基幹手法。

### 1.3 罠（PFの尤度設計に直結する重要点 → 打ち手C）
GRは万能ではない。次のケースで「岩相 ↔ GR」が壊れる：
- **K長石・雲母・グローコナイトを含む清浄砂岩** → GRが高く出てシェールに誤認される。
- **ウラン異常（有機物に伴うU濃集）** → 清浄砂岩がシェール並みのGRを示す。スペクトルGRでないと分離不能。
- **"hot" dolomite（特にPermian Basin）** → Uに富み、最大200 APIに達してシェールと区別がつかない。
- **放射性カリ泥(KCl mud)** → ベースライン全体が ~20 API 底上げされる（**baseline drift**）。

> **CVへの含意:** あなたの分析メモにある「GR baseline drift が今のPFの主要ノイズ源」は、まさにこの物理に対応する。
> 生GR差分を尤度に使うと、well毎のbaseline drift（KCl mud, ツール較正差）がそのまま誤差になる。
> **well毎に正規化したGR（z-score / median引き）や微分（dGR/dMD）を尤度に使う**のは、物理的に正しい。
> drift は加法的なので、差分（微分）を取れば原理的に消える。

### 1.4 GR曲線の「形」が語る堆積環境（marker bed 同定 → 打ち手C）
業界標準（Rider のlog shape分類）：
- **Funnel shape（クリーンアップ／上方粗粒化）**: GRが上方に向かって減少。砂質化・浅海化。
- **Bell shape（ダーティアップ／上方細粒化）**: GRが上方に向かって増加。流路・潮汐チャネル。
- **Serrated / blocky / irregular**: 薄互層、アグラデーション。
- **Maximum flooding surface**: U peak (>5 ppm) かつ低Th/U比 (<2.5) — スペクトルGRがあれば強力なマーカー。

> **CVへの含意:** geosteererが実際にやるのは「GR波形全体のマッチ」ではなく **特徴的なピーク／形のtying（マーカー一致）**。
> PFの尤度を「全点のGR残差」から「**マーカー（特徴点）の一致度**」に寄せると、47壊れwellを救える可能性。
> 具体的には、GRの極大/極小・勾配反転点を特徴点として抽出し、その並びの一致を重み付けする設計。

---

## 2. 幾何 — TVD / TVT / TST とdip（打ち手Bの理論的核心）

このコンペのターゲット `TVT` の意味を、業界定義で厳密に押さえる。3つの「厚み」は混同しやすいので図式で。

### 2.1 三者の定義（AAPG Wiki / SPWLA 準拠）
- **TVD (True Vertical Depth)**: 各点を「鉛直に掘ったら何ftか」に換算した深度。地図化の基本だが、高傾斜井では不正確。
- **TVT (True Vertical Thickness)**: 地層単位を **鉛直方向に測った厚み**。
  - 重要性質: **dipの変化に影響されにくく**、構造ホライズンを引き算すれば求まる。体積計算に有効。
  - → だから **ターゲットに選ばれている**（=構造に対して安定した不変量に近い）。
- **TST (True Stratigraphic Thickness)**: 地層面に**垂直**に測った真の地層厚。最も本質的な地質量。

### 2.2 幾何関係（dip = 地層傾斜角 を介した変換）
平面地層・偏距のある坑井で、地層傾斜 δ（dip）を介して：

```
TST = TVT × cos(δ)
（地層が水平 δ=0 なら TVT = TST。dipが大きいほど両者は乖離する）
```

- **dip（真の傾斜）** は走向(strike)に直交する方向の最大傾斜。
- **apparent dip（見かけ傾斜）** は任意断面方向で見た傾斜で、常に真dip以下。
  - 断面が走向に直交（dip方向）のときだけ真dip = apparent dip。
  - **CVへの含意:** 水平井の進行方位が dip方位とずれていると、断面上のdipは見かけ値。あなたの Group F（dTVT/dZ）が1次近似に留まるのはここ。本来は **dip + azimuth（roll）の2成分**で扱うのが業界標準。

### 2.3 業界の核心ワークフロー「stretch & invert」（打ち手A/D/Bすべての母体）
URTeC 1590259 等が定式化した、水平井geosteeringの手作業の中身：

1. 近傍の縦井 = **typewell（pilot/template log）** を用意。
2. 水平井のGRを、typewellのGRプロファイル上で **squeeze（圧縮）/ stretch（伸長）** する。
   - 圧縮が要る理由: typewell側の薄い層を、水平井は長い距離をかけて貫くため。
3. **invert（反転）** も許す。
   - 反転が要る理由: ドリルは地層を **下→上にも上→下にも** 横切るため（up-section / down-section）。
4. こうして得た整合から **apparent dip のブロック列** を求め、bit のstratigraphic positionを決める。

> これは数学的には **「dip補正つきDTW（Dynamic Time Warping）」** そのもの。
> あなたのメモの「**slide & stretch = DTW + dip補正**、exp020 soft DTW は失敗したが well単位ハード + dip制約付きは未試行**」は、業界手法と完全に一致した洞察。
> 参照解法のBeam Search（DTW型14config）は、この「stretch & invert」を離散探索で再現したもの。

### 2.4 TST/TSP は「look ahead」できる（打ち手Bの精度向上）
URTeC論文の主張: TST(TSP)表示は **bitの先（look ahead）** を読める点でTVD表示より優れる。
- **CVへの含意:** 状態を `(TVT, dTVT/dMD)`（=位置 + 構造傾斜rate）まで持ち、build区間で較正したdipを事前分布に入れる、という打ち手B高次化は、TSTの「look ahead」性を確率フィルタで再現すること。dipが滑らかに連続する前提が地質的に妥当（断層がなければ）。

---

## 3. typewell と「割り当て問題」（打ち手B = 最大の伸びしろ）

### 3.1 typewellとは
予測対象の水平井に対し、**正解パターンを与える参照断面**。通常は近傍の縦井（pilot well）のGR-深度プロファイル。
あなたの現PFは **1 well = 1 typewell** 前提。

### 3.2 ROGII公式が示す「正解の方向」
StarSteer / GeoAssist の公式記述から読み取れる設計思想（＝あなたの未実装方向）：
- 「**自動で最も近いtypewellを見つけ、wellをlandさせ、geosteering解釈を提供する**」（multi-well auto geosteering）。
- 「**1つのlateral wellに対し複数のoffset wellから複数の解釈を構築し、最良の相関を見つけて比較する**」。
- 「strat-based と model-based の両アプローチで、**適切なdip・断層・層厚変化**を持つモデルを作る」。
- GeoAssist の ML は「**1000s of wells across numerous reservoirs**」で学習。

> **CVへの含意（核心）:** 「**47壊れwell = 割り当てtypewellが悪い**」という仮説は、ROGIIの設計思想が裏付ける。
> 業界ソフトすら「1 wellに複数typewellを試して最良相関を選ぶ」のだから、1 well固定は明確な過小利用。
> **multi-hypothesis PF**（各水平井に top-K typewell候補を選び、独立にPFを走らせ尤度加重平均）は、
> ROGIIの「複数offset wellから最良相関」をそのまま確率フィルタ化したもの。実装難度も低く、最も費用対効果が高い。

### 3.3 typewell選択の地質的基準（top-K候補の選び方）
単なる空間距離だけでなく、以下が近いtypewellを優先すべき：
- **GRエネルギー/レンジが似ている**（同じ堆積相・同じbaseline）。
- **既知区間（build/landing）でのGR波形相関が高い**。
- **構造的に同じブロック**（断層で隔てられていない）。

---

## 4. PNG構造断面図に何が描かれているか（打ち手A = 最大の伸びしろ・要確認）

### 4.1 業界での「断面図」の種類
- **Vertical section（鉛直断面）**: dip方位が断面外だとapparent dipやhorizonの表示が崩れる（業界既知の問題）。
- **Curtain section / displacement section（カーテンセクション）**: 坑跡に沿った「垂れ幕」状の断面。**dipとhorizonを正しく表示できる**ため geosteering標準。複数の水平井を共通のcurtain sectionに重ねることも行う。

### 4.2 典型的に描かれる要素（文献の実例より）
geosteering解釈の断面図には一般に：
- **top / bottom horizon**（地層境界の解釈線）。
- **target zone**（薄いペイ層、しばしば10ft級の narrow window）。
- **well path（坑跡）** と、その上に重ねた **GR/typelog**。
- geosteerer の解釈点 = しばしば **「typelog（青線）上に重ねた赤いドット列」** として表現される（文献の実例図そのもの）。

> **CVへの含意（暴力的に強い可能性）:** もしPNGに **解釈済みhorizon線やtarget境界** が描かれていれば、
> それは **正解のTVT境界がほぼ与えられている** に等しい。CNN/画像処理で線を抽出 → TVT-on-MD regression にすれば、
> PFと**直交する独立シグナル**になり、blendで大きく効く（メモの「当たれば −1.0以上は射程」と整合）。
> **まず1枚開いて、構造線が機械可読な形で描かれているか確認する**のが最優先（5分で判定可能）。
> 注意: 学習画像に正解horizonが描かれていて、それがtest画像にも同等に描かれていれば信号。
> test画像が「解釈前」なら使えない。**train/test両方のPNGを1枚ずつ比較**して、情報の対称性を必ず確認すること。

---

## 5. アルゴリズムの系譜 — 参照解法を業界文脈に位置づける（打ち手D）

### 5.1 geosteering inversion の状態空間
学術・特許文献での定式化（あなたのPFと同型）：
- 状態 = formation boundary の位置（= TVT）と dip。
- **Recursive Bayesian filter（逐次ベイズフィルタ）= Particle Filter** が「sequential estimation」を解く標準手法。
- 対して **larger-scale inference（全域最適）** を解く系統もある（DP / Viterbi 的）。
  - → あなたのメモの「**DPトラッカーをPFと並列に走らせblend**（PFの迷子を全域探索で抑える）」は、文献上もPFと相補的な正攻法。

### 5.2 米国特許に見る「physical model」の実体（打ち手D = exp023再特定）
US10318662 / US20130238306 等のgeosteering特許の定式化：
- 2Dモデル（直線近似）の **dip または fault offset を調整** し、
- **予測LWD曲線と実測LWD曲線**（または実測TVT曲線と予測TVT typelog）が一致するまで反復、
- 一致区間の端に **marker をセット**して次区間へ。

> **CVへの含意:** 参照ノートの「physical model」分岐は、単純なleak lookup（真TVT参照）か、
> あるいはこの **dip+fault offsetをフィットする構造モデル** か、どちらか。後者なら汎化する正攻法。
> exp023はlookupと結論したが、**dip/fault offsetフィット型の構造モデルの可能性**を再精査する価値がある（メモのD項と一致）。

### 5.3 最新手法（2024–2025）
- **U-Net系で horizon auto-tracking**（seismic/断面画像のセグメンテーション）。打ち手AのCNN regressionと同系統。
- **3D horizon tracking × log interpretation のハイブリッド**で near-wellbore formation shapeを再構成（MDPI 2024）。
- **DTW を seismic trace相関に使う**手法（Hale 2013ほか）。GR相関へのDTW適用は十分に確立。

---

## 6. 打ち手 A〜D とドメイン知識の対応表

| 打ち手 | 中身 | 支えるドメイン知識（本レポート該当節） | 期待効果 |
|---|---|---|---|
| **A** | PNG断面図 → TVT-on-MD regression | §4 curtain section / horizon / target zone の表現様式 | 当たれば −1.0以上、PFと直交 |
| **B-1** | multi-typewell PF（top-K加重平均） | §3 ROGII設計思想・typewell割り当て / §1.4 marker | 47壊れwell救出、pooled CV大幅減 |
| **B-2** | 状態を (TVT, dTVT/dMD) に高次化 | §2.2–2.4 dip+azimuth / TSTのlook ahead | geom offset bias低減 |
| **C-1** | 尤度を well正規化GR / 微分に | §1.3 baseline drift の物理 | PF主要ノイズ源を除去 |
| **C-2** | 尤度をmarker一致度に | §1.4 log shape・MFS・特徴点tying | broken well救出 |
| **D** | 参照「physical model」再特定 | §5.2 dip+fault offsetフィット特許 | 正攻法の構造モデル発見の可能性 |

---

## 7. 次の一手（ドメイン視点での推奨順）

1. **PNGを train/test 各1枚開く（§4）** — 構造線・target境界が機械可読か、train/testで情報が対称かを判定。5分。当たればコンペ構造ごと書き換わる。
2. **multi-typewell PF（§3）** — ROGII設計思想に最も忠実。47壊れwellの大半を救える見込み。実装1日。
3. **尤度の物理的見直し（§1.3, §1.4）** — 生GR差分 → well正規化GR/微分 + marker特徴点。PFの土台を地質的に正す。
4. **DPトラッカー並列 + blend（§5.1）** — PFの迷子に対する保険。
5. **参照physical model再精査（§5.2）** — dip+fault offsetフィット型なら汎化する正攻法。

> チューニング系（PF param微調整・blend重み）は誤差範囲なので打ち切り、という判断はドメイン視点でも妥当。
> 伸びしろは「**未活用シグナル（PNG）**」と「**typewell割り当ての是正**」にある。

---

## 用語ミニ辞書

- **GR (Gamma Ray)**: 自然ガンマ線検層。泥質で高く、砂岩/炭酸塩で低い。
- **API (gAPI)**: GRの単位。
- **TVD / TVT / TST**: 鉛直深度 / 鉛直層厚 / 地層垂直層厚。`TST = TVT·cos(dip)`。
- **dip / apparent dip**: 真の地層傾斜 / 任意断面で見た見かけ傾斜（≤真dip）。
- **strike（走向）**: dip方向に直交する水平線の方位。
- **typewell / pilot / template log**: 正解パターンを与える参照（縦）井。
- **typelog**: typewellのGRプロファイル（断面図で青線、解釈点が赤ドットで重なる）。
- **stretch & invert (squeeze & flip)**: 水平井GRをtypelogに整合させる伸縮+反転 = dip補正DTW。
- **TSP (True Stratigraphic Position)**: log相関でapparent dipを求める地層位置決定法。TSTと近縁。
- **curtain / displacement section**: 坑跡に沿う断面。dipとhorizonを正しく表示。geosteering標準。
- **landing**: 水平区間に入る前にbitをtarget層へ着地させること。
- **target zone / pay**: 目標とする薄いペイ層（しばしば~10ft）。
- **LWD (Logging While Drilling)**: 掘削同時検層。水平井のGRはこれ。
- **recursive Bayesian filter**: 逐次ベイズ推定 = Particle Filter の母体。
- **horizon auto-tracking**: 地層境界面を自動追跡する数値手法（U-Net等）。
- **StarSteer / GeoAssist (ROGII)**: 本コンペ主催ROGIIのgeosteering / ML自動解釈ソフト。
