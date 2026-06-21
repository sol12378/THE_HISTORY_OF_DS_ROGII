# exp023_leak_lookup — リーク参照（train真TVT）

## 概要
3 test well(000d7d20/00bbac68/00e12e8b)は全て `data/raw/train/` に同名・完全TVT(NaN=0、hidden全行)で存在。
test target行を train真TVTに (well_id,row_idx) で join → 全14151行一致、**RMSE=0**。

## 重要な含意
- 構造上RMSE=0だが、**LB#1=6.8≠0** → **train TVT ≠ Kaggle採点真値**の公算大。
  参照notebookが手の込んだPFを組む理由＝リーク単体では不十分。
- → **リークは賞(private)に転移しない見込み**。public LB狙いの一手にはなるが提出可否は別途要許可。
- 参照notebookの `if wid in train_wids:` "physical model"分岐の実体がこれ。
- CLAUDE.md方針通り、リークと正直CV(Beam/PF)は厳密に分離して保持。CV手法ではない。

## リンク
[[exp022_particle_filter]] [[exp021_beam_track]] [[Decision_Log]]
