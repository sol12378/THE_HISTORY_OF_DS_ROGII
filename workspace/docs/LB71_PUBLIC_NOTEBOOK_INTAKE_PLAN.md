# LB 7.1 public notebook intake plan

Date: 2026-06-21

## Reference notebooks

The submission floor is now the LB 7.1 public-notebook family. New candidates should not be promoted unless they plausibly compete with this family, with `lightningv08/rogii-lb-7-168` as the current best public reference.

| Exp | Kernel | Reported LB | Local directory | Status |
|---|---|---:|---|---|
| exp090 | lightningv08/rogii-lb-7-168 | 7.168 | `workspace/kaggle_notebooks/exp090_lightningv08_lb7168` | pulled/extracted |
| exp091 | baidalinadilzhan/rogii-lb-7-201 | 7.201 | `workspace/kaggle_notebooks/exp091_baidalinadilzhan_lb7201` | pulled/extracted |
| exp092 | kokinnwakashuu/rogii-dual-pipeline-v16 | 7.1 vicinity | `workspace/kaggle_notebooks/exp092_kokinnwakashuu_dual_pipeline_v16` | pulled/extracted |
| exp093 | curvecowboy/rogii-lb7295-public-rebuild-submit | 7.295 | `workspace/kaggle_notebooks/exp093_curvecowboy_lb7295_public_rebuild` | pulled/extracted |
| exp094 | omprakashpy/rogii-public-gold-fallback | 7.1 vicinity | `workspace/kaggle_notebooks/exp094_omprakashpy_public_gold_fallback` | pulled/extracted |

## Initial source grouping

The five notebooks collapse to three source families by script hash.

| Group | Members | Interpretation |
|---|---|---|
| A | exp090, exp093 | LB 7.168 / public rebuild source are identical at script level |
| B | exp091, exp094 | LB 7.201 / public gold fallback source are identical at script level |
| C | exp092 | dual-pipeline-v16 is the distinct richer variant |

## Intake gates

1. Pull notebook source into the registered directory.
2. Record kernel metadata, dataset sources, model/package sources, and internet/GPU flags.
3. Classify leakage risk: public overlap use, train-only formation columns, hidden target use, external artifact provenance.
4. Rebuild a local submission and validate shape with the shared submission gate.
5. Compare against exp080 and exp090 references before any Kaggle submission.

## Acquisition notes

Python HTTPS in this Windows environment currently fails with `OPENSSL_Applink`, and curl initially fails revocation checks. The working acquisition path is Kaggle legacy `/kernels/pull` through `curl.exe --ssl-no-revoke`, followed by `scripts/extract_kaggle_kernel_pull.py`.
