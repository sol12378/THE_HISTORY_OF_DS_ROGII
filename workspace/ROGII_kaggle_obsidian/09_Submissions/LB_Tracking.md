# LB Tracking

| Date | Submission | Source Exp | CV | LB | Gap | Notes |
|---|---|---|---:|---:|---:|---|
| 2026-05-25 | `submission.csv` | [[exp003_lgb_anchor_trajectory]] | 15.054865 | 14.147 | -0.908 | Kaggle Notebook version 3から提出。Notebook-only competitionのためCSV直接提出ではなくcode submissionを使用。 |
| 2026-06-07 | exp072_proj | [[exp068]] | 9.086 | **8.280** | 0.806 | 現best LB。exp051 blend + projection。 |
| 2026-06-10 | exp073_blend | [[exp073_public_assets_integration]] | 8.79 | 8.630 | +0.16 | **棄却(LB悪化)**。借り物外部OOF膨張で負の転移。確定best=exp072 8.280だった |
| 2026-06-11 | exp080 sp45-fleongg fork | aiwody fork | - | **7.625** | - | **新best!** 公開sp45-fleongg-blend-v2をclean fork(公開datasetのみ、overlap-override等の水増し無し)。従来best 8.280から-0.655。次のbase |
