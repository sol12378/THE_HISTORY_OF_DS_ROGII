#!/usr/bin/env python
"""Phase 1: drift-well forensics on exp073 blend OOF.
Per-well error census + failure-mode characterization to aim Phase 2 (MHT tracker).
Leak-free diagnostic (reads OOF preds + geometry only).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"E:\kaggle\THE_HISTORY_OF_DS_ROGII")
OUT = ROOT / "experiments" / "exp073_public_assets_integration"
TB = ROOT / "data" / "processed" / "train_base_v001.parquet"


def well_rmse(d):
    return np.sqrt(np.mean(d ** 2))


def main():
    blend = pd.read_csv(OUT / "oof_blend_final.csv")  # id, well_id, tvt_true, pred_tvt
    pf = pd.read_csv(OUT / "oof_pf.csv", usecols=["id", "pred_tvt"]).rename(columns={"pred_tvt": "pf"})
    geom = pd.read_csv(OUT / "oof_geom.csv", usecols=["id", "pred_tvt"]).rename(columns={"pred_tvt": "geom"})
    df = blend.merge(pf, on="id").merge(geom, on="id")

    tb = pd.read_parquet(TB, columns=["well_id", "row_idx", "MD", "Z", "GR", "is_known_tvt",
                                       "last_known_TVT", "last_known_MD"])
    tb["id"] = tb["well_id"] + "_" + tb["row_idx"].astype(str)
    hid = tb[~tb["is_known_tvt"]]
    df = df.merge(hid[["id", "MD", "Z", "GR", "last_known_MD"]], on="id", how="left")

    df["err_blend"] = df["pred_tvt"] - df["tvt_true"]
    df["err_pf"] = df["pf"] - df["tvt_true"]
    df["err_geom"] = df["geom"] - df["tvt_true"]

    # per-well census
    rows = []
    for wid, g in df.groupby("well_id"):
        rb = well_rmse(g["err_blend"].values)
        rp = well_rmse(g["err_pf"].values)
        rg = well_rmse(g["err_geom"].values)
        rows.append({
            "well_id": wid, "n_hidden": len(g),
            "rmse_blend": rb, "rmse_pf": rp, "rmse_geom": rg,
            "z_span": float(g["Z"].max() - g["Z"].min()),
            "md_span": float(g["MD"].max() - g["last_known_MD"].iloc[0]),
            "gr_std": float(np.nanstd(g["GR"].values)),
            "bias_blend": float(g["err_blend"].mean()),
        })
    w = pd.DataFrame(rows)
    n = len(w)

    def band(col):
        good = (w[col] < 8).sum(); bad = ((w[col] >= 8) & (w[col] < 20)).sum(); brk = (w[col] >= 20).sum()
        ss = (w[col] ** 2 * w["n_hidden"]).sum()
        contrib = {
            "good_<8": round(float((w.loc[w[col] < 8, col] ** 2 * w.loc[w[col] < 8, "n_hidden"]).sum() / ss), 3),
            "bad_8-20": round(float((w.loc[(w[col] >= 8) & (w[col] < 20), col] ** 2 * w.loc[(w[col] >= 8) & (w[col] < 20), "n_hidden"]).sum() / ss), 3),
            "broken_>20": round(float((w.loc[w[col] >= 20, col] ** 2 * w.loc[w[col] >= 20, "n_hidden"]).sum() / ss), 3),
        }
        pooled = float(np.sqrt(ss / w["n_hidden"].sum()))
        return good, bad, brk, contrib, pooled

    print("=== per-well census (n=%d wells) ===" % n)
    for col in ["rmse_blend", "rmse_pf", "rmse_geom"]:
        good, bad, brk, contrib, pooled = band(col)
        print(f"{col}: pooled={pooled:.3f} | good<8={good} bad8-20={bad} broken>20={brk} | pooled^2 contrib={contrib}")

    # broken wells of the BLEND
    brokenw = w[w["rmse_blend"] >= 20].sort_values("rmse_blend", ascending=False)
    print(f"\n=== BLEND broken wells (>20ft): {len(brokenw)} ===")
    print(brokenw[["well_id", "n_hidden", "rmse_blend", "rmse_pf", "rmse_geom", "z_span", "md_span", "gr_std", "bias_blend"]].head(25).to_string(index=False))

    # how many broken are 'pf broke it' vs 'geom broke it' vs both
    bb = w[w["rmse_blend"] >= 20]
    print(f"\nBroken breakdown: pf_also_broken={int((bb['rmse_pf']>=20).sum())}/{len(bb)}  geom_also_broken={int((bb['rmse_geom']>=20).sum())}/{len(bb)}")
    # wells where pf broken but geom good (geom could rescue) and vice versa
    rescue_geom = int(((w["rmse_pf"] >= 20) & (w["rmse_geom"] < 12)).sum())
    rescue_pf = int(((w["rmse_geom"] >= 20) & (w["rmse_pf"] < 12)).sum())
    print(f"pf broken but geom<12 (geom rescues): {rescue_geom}")
    print(f"geom broken but pf<12 (pf rescues): {rescue_pf}")

    # correlation of broken-ness with features
    print("\n=== broken correlates (spearman rmse_blend vs feature) ===")
    for f in ["n_hidden", "z_span", "md_span", "gr_std"]:
        c = w[["rmse_blend", f]].corr(method="spearman").iloc[0, 1]
        print(f"  {f}: {c:+.3f}")

    # oracle: if we could pick min(pf,geom,blend) per well
    w["rmse_oracle_pick"] = w[["rmse_blend", "rmse_pf", "rmse_geom"]].min(axis=1)
    oracle_pooled = float(np.sqrt((w["rmse_oracle_pick"] ** 2 * w["n_hidden"]).sum() / w["n_hidden"].sum()))
    print(f"\noracle per-well pick min(blend,pf,geom): pooled={oracle_pooled:.3f}  (blend pooled={band('rmse_blend')[4]:.3f})")

    w.sort_values("rmse_blend", ascending=False).to_csv(OUT / "forensics_per_well.csv", index=False)
    summary = {
        "n_wells": n,
        "blend_pooled": band("rmse_blend")[4],
        "broken_count": int((w["rmse_blend"] >= 20).sum()),
        "bad_count": int(((w["rmse_blend"] >= 8) & (w["rmse_blend"] < 20)).sum()),
        "broken_pooled2_contrib": band("rmse_blend")[3]["broken_>20"],
        "bad_pooled2_contrib": band("rmse_blend")[3]["bad_8-20"],
        "rescue_geom_for_pf_broken": rescue_geom,
        "oracle_per_well_pick": oracle_pooled,
    }
    json.dump(summary, open(OUT / "forensics_summary.json", "w"), indent=2)
    print("\nsaved forensics_per_well.csv + forensics_summary.json")


if __name__ == "__main__":
    main()
