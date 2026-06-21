# exp034_hybrid_leak — Leak転移判定 (棄却確定)

## 概要
exp033 (8-component blend) 50% + exp023 (train真TVT lookup) 50% の hybrid を Kaggle kernel化、自己完結で再計算submit。leak がpublic LBに転移するかを実測。

## 構成
```
final = 0.5 * model_blend + 0.5 * leak_lookup
model_blend = NNLS(pf_tuned 0.51 + pf_multi 0.29 + pf_phys 0.15 + geom 0.05)
leak_lookup = train CSV の TVT列 を test wellsから直接 lookup
```

multiprocessing PF (3 wells), LightGBM 5-fold geom, smoothing (w=101)。
Kernel: sol12378/rogii-exp034-hybrid-leak。runtime ~45min on Kaggle CPU。

## 結果 — **Leak転移しない確定**

| | LB |
|---|---:|
| exp026 (model only, LB既知) | **8.672** |
| **exp034 (0.5 model + 0.5 leak)** | **10.794** (+2.12悪化) |

**leak成分は採点真値から大幅に乖離**。0.5重みでもLBを悪化させる。

### Leak単体LB推定
- corr=0仮定: pure leak LB ≈ 19.8
- corr=1仮定: pure leak LB ≈ 12.9
- 実態は中間で **13〜19** と推定

## 含意

1. **train TVT ≠ test採点真値** (構造的に異なる、ノイズではない)
2. **SaintLouis LB 5.986は leak ではない別手段**
   - MTP-CNN (Alyaev 2022)?
   - PluRaListic RL+PF?
   - External well datasets (ROGII公開分)?
   - 隠れた別系統leak?
3. exp037 w01-w10 (leak weight binary search 5 kernel) は **全て submit禁止**
4. 戦略: **leak-free路線に全振り**
5. **exp036 (safe baseline) を提出**して exp033相当のLB測定

## 決定

- Decision_Log 2026-06-05: leak利用完全棄却
- 残りsubmission枠: exp036 (safe baseline), exp035 (TabICL完了後), 新規手法

## リンク
[[exp026_final_blend]] [[exp033_final_blend]] [[exp023_leak_lookup]]
