# LB Tracking

| Date | Submission | Source Exp | CV | LB | Gap | Notes |
|---|---|---|---:|---:|---:|---|
| 2026-05-25 | `submission.csv` | [[exp003_lgb_anchor_trajectory]] | 15.054865 | 14.147 | -0.908 | Kaggle Notebook version 3から提出。Notebook-only competitionのためCSV直接提出ではなくcode submissionを使用。 |
| 2026-06-07 | exp072_proj | [[exp068]] | 9.086 | **8.280** | 0.806 | 現best LB。exp051 blend + projection。 |
| 2026-06-10 | exp073_blend | [[exp073_public_assets_integration]] | 8.79 | 8.630 | +0.16 | **棄却(LB悪化)**。借り物外部OOF膨張で負の転移。確定best=exp072 8.280だった |
| 2026-06-11 | exp080 sp45-fleongg fork | aiwody fork | - | **7.625** | - | **新best!** 公開sp45-fleongg-blend-v2をclean fork(公開datasetのみ、overlap-override等の水増し無し)。従来best 8.280から-0.655。次のbase |
| 2026-06-21 | exp090 lightningv08 public reference | lightningv08/rogii-lb-7-168 | - | **7.168** | - | 新しい最低提出下限。LB 7.1台公開notebook群の主基準。source pulled/extracted |
| 2026-06-21 | exp091 baidalinadilzhan public reference | baidalinadilzhan/rogii-lb-7-201 | - | **7.201** | - | LB 7.1台公開notebook群の比較基準。source pulled/extracted |
| 2026-06-21 | exp092 dual-pipeline-v16 | kokinnwakashuu/rogii-dual-pipeline-v16 | - | 7.1付近 | - | dual pipeline系の比較基準。source pulled/extracted |
| 2026-06-21 | exp093 public rebuild | curvecowboy/rogii-lb7295-public-rebuild-submit | - | **7.295** | - | public rebuild/fallback系の比較基準。exp090とscript同一 |
| 2026-06-21 | exp094 public gold fallback | omprakashpy/rogii-public-gold-fallback | - | 7.1付近 | - | gold fallback系の比較基準。exp091とscript同一 |
| 2026-06-21 | exp096 clean LB7 base | [[exp096_clean_lb7_base]] (exp090 fork) | - | **7.297** | - | **新・自前best(従来7.625を−0.328更新)**。exp090の2段public-overlap leak(cell37本体+cell38 gold)を両方no-op化したclean版(`rows overridden=0`確証)。**leak付き公開7.168との差はわずか0.13**=leakはほぼ見かけだけ・Private非転移。これがPrivate転移する確定下限base。 |
