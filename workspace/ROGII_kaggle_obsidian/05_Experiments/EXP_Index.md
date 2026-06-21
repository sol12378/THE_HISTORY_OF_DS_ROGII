# 実験インデックス

| Exp | Status | Model | Features | CV | LB | Notes |
|---|---|---|---|---:|---:|---|
| exp000_leak_lookup | planned | lookup | leaked train target | - | - | 提出形式確認のみ |
| exp001_anchor_baseline | completed | rule | anchor | 15.909853 | - | [[exp001_anchor_baseline]] |
| exp002_slope_baseline | planned | rule | anchor+slope | - | - | trajectory sanity check |
| exp003_lgb_anchor_trajectory | completed | LightGBM | anchor+trajectory+GR | 15.054865 | - | [[exp003_lgb_anchor_trajectory]] |
| exp004_oof_error_slicing | completed | analysis | OOF slices | 15.054865 | 14.147 | [[exp004_oof_error_slicing]] |
| exp005_worst_well_analysis | completed | analysis | worst well pattern | 15.054865 | 14.147 | [[exp005_worst_well_analysis]] |
| exp006_anchor_guard | completed | post-process | anchor guard B案 | 14.724541 | - | [[exp006_anchor_guard]] |
| exp007_traj_features | completed | LightGBM | SAFE+Group A+B+C | 13.867054 | - | [[exp007_traj_features]] |
| exp007b_guard | completed | LGB(n=1500)+guard | SAFE+Group A+B+C | 13.853360(raw) | - | [[exp007b_guard]] |
| exp008_gr_rolling | completed | LightGBM | SAFE+Group A+B+C+D | 13.808621 | - | [[exp008_gr_rolling]] |
| exp009_typewell_gr_alignment | completed | LightGBM | SAFE+Group A+B+C+D+E | 13.922291 | - | [[exp009_typewell_gr_alignment]] |
| exp009b_typewell_gr_well_only | completed | LightGBM | SAFE+A+B+C+D+E_well | 13.932154 | - | [[exp009b_typewell_gr_well_only]] |
| exp010_oof_slicing_fold24 | completed | analysis | OOF slices (corr層別) | 13.808621 | - | [[exp010_oof_slicing_fold24]] |
| exp011_typewell_leak_test | completed | LGB×2 (DiD) | E無し vs E有り on dedup fold | base 13.651645 / E 13.928077 | - | [[exp011_typewell_leak_test]] |
| exp012_anchor_guard_exp008 | completed | post-process | exp008 + anchor guard | 13.754461 (fold非一貫) | - | [[exp012_anchor_guard_exp008]] |
| exp013_gr_match_go_nogo | completed | rule(leak-free) | GR-typewell直接照合 | A=43.08/anchor15.91 | - | [[exp013_gr_match_go_nogo]] |
| exp013b_gr_local_refine | completed | rule(leak-free) | GR局所微修正 | best 13.826>exp008 NO-GO | - | [[exp013b_gr_local_refine]] |
| exp014_geom_extrap | completed | LightGBM | SAFE+A+B+C+D+**F**幾何外挿 | **13.525189** (+0.283) | - | [[exp014_geom_extrap]] |
| exp015_seq_smooth | completed | post-process | well内 mean平滑 w=101 | **13.520383** (現best) | - | [[exp015_seq_smooth]] |
| exp016_struct_plane | completed | LightGBM | +G 3D構造平面外挿 | 13.887794 (棄却 -0.363) | - | [[exp016_struct_plane]] |
| exp021_beam_track | completed | Beam Search(DTW型) | GR-typewell系列照合 | 15.697 (≈anchor, 失敗) | - | [[exp021_beam_track]] |
| exp022_particle_filter | completed | Particle Filter | GR-typewell尤度トラッキング | **11.024 (最良単体)** | - | [[exp022_particle_filter]] |
| exp023_leak_lookup | completed | リーク参照 | train真TVT lookup | 0.000 (賞転移せず見込み) | - | [[exp023_leak_lookup]] |
| (PF×geom blend) | analysis | post(0.6PF+0.4geom)+平滑 | — | 10.16 | - | [[exp022_particle_filter]] |
| exp024_multistage_blend | completed | NNLS blend+平滑 | PF×geom×trees×nn×attn | **10.077** | - | [[exp024_multistage_blend]] |
| exp025_pf_tune | completed | PF param grid(subset) | init_spread/PN/scale | subset best 10.468 | - | [[exp025_pf_tune]] |
| exp025_pf_tuned (full) | completed | tuned PF full773 | init_spread=4,PN=0.01 | 10.984 | - | [[exp025_pf_tune]] |
| exp026_final_blend | completed | NNLS blend+平滑 | tunedPF×geom×trees×nn×attn | **10.062 (現best)** | - | [[exp026_final_blend]] |
