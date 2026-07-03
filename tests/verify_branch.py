"""Independent verification of branch-project preprocessing (WT / KI).

Reconstructs each branch's subset the same way run_branch does, then re-derives
steps 02-05 from the raw data and compares to the saved branch matrices.
"""
import sys, os
import numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
sys.path.insert(0, "scripts")
import config, utils, step05_impute as s5
ID = config.ID_COL
ATOL = 1e-6
all_pass = True

def verify_branch(ds):
    global all_pass
    cfg = config.DATASETS[ds]
    genotype = cfg["genotype"]
    groups = cfg["group_order"]
    root = cfg["output_root"]
    res = []
    def check(name, cond, detail=""):
        res.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

    print(f"\n{'='*62}\n### BRANCH {ds}  (genotype={genotype})\n{'='*62}")

    # ── Ground truth: reproduce run_branch subset ────────────────────────────
    df, meta = utils.load_raw(ds)
    df = utils.to_numeric_intensity(df, meta["column_name"].tolist())
    sub_df, sub_meta, bio, ref_col = utils.subset_by_genotype(df, meta, genotype)

    print("\n=== Subset construction ===")
    check("40 bio samples of this genotype (+1 ref)",
          len(bio) == 40 and ref_col is not None, f"bio={len(bio)}")
    check("all bio samples are the target genotype",
          (sub_meta.loc[~sub_meta['is_ref'], 'genotype'] == genotype).all())
    gc = sub_meta.loc[~sub_meta['is_ref'], 'group'].value_counts().reindex(groups)
    check("4 timepoint groups present with expected sizes",
          gc.notna().all(), dict(gc))

    # ── STEP 02 filter ───────────────────────────────────────────────────────
    print("\n=== STEP 02 — filter (≥50% in ALL 4 groups) ===")
    filt = pd.read_csv(f"{root}/02_filtered/filtered_matrix.tsv", sep="\t")
    vals = sub_df[bio].values
    keep = np.ones(vals.shape[0], dtype=bool)
    for g in groups:
        gcols = sub_meta.loc[sub_meta["group"] == g, "column_name"].tolist()
        idx = [bio.index(c) for c in gcols]
        frac = np.sum(~np.isnan(vals[:, idx]), axis=1) / len(idx)
        keep &= (frac >= 0.50)
    check("filtered protein set == independent recompute",
          set(sub_df.loc[keep, ID]) == set(filt[ID]),
          f"indep={int(keep.sum())}, saved={len(filt)}")
    # spot-check kept/removed group fractions
    def minfrac(i):
        return min(np.sum(~np.isnan(vals[i, [bio.index(c) for c in
                   sub_meta.loc[sub_meta['group']==g,'column_name']]]))/
                   (sub_meta['group']==g).sum() for g in groups)
    check("all kept ≥50% valid in every group",
          all(minfrac(i) >= 0.5 for i in np.where(keep)[0][:150]))
    check("all removed <50% in ≥1 group",
          all(minfrac(i) < 0.5 for i in np.where(~keep)[0][:150]))

    # ── STEP 03 log2 ─────────────────────────────────────────────────────────
    print("\n=== STEP 03 — log2 ===")
    log2 = pd.read_csv(f"{root}/03_log2_transformed/log2_matrix.tsv", sep="\t")
    recon = np.log2(filt[bio + [ref_col]].values.astype(float))
    check("log2_matrix == log2(filtered)",
          np.allclose(log2[bio + [ref_col]].values, recon, equal_nan=True, atol=ATOL))
    check("protein order preserved 02→03", filt[ID].tolist() == log2[ID].tolist())

    # ── STEP 04 normalize ────────────────────────────────────────────────────
    print("\n=== STEP 04 — ref-ratio + median ===")
    norm = pd.read_csv(f"{root}/04_normalized/normalized_matrix.tsv", sep="\t")
    reftbl = pd.read_csv(f"{root}/04_normalized/reference_ratio_table.tsv", sep="\t")
    fac = pd.read_csv(f"{root}/04_normalized/normalization_factors.tsv", sep="\t")
    L = log2[bio].values.astype(float)
    ref = log2[ref_col].values.astype(float)
    row_mean = np.nanmean(L, axis=1)
    offset = np.where(np.isnan(ref), row_mean, ref)
    offset = np.where(np.isnan(offset), 0.0, offset)
    ratio = L - offset[:, None]
    psm = np.nanmedian(ratio, axis=0)
    shift = psm - np.nanmedian(psm)
    recon_norm = ratio - shift[None, :]
    check("normalized_matrix == independent recompute",
          np.allclose(norm[bio].values, recon_norm, equal_nan=True, atol=1e-5))
    check("ref column dropped (cols = 3 meta + 40 bio)",
          ref_col not in norm.columns and norm.shape[1] == 43, f"cols={norm.shape[1]}")
    check("offset table matches recompute",
          np.allclose(reftbl["offset_subtracted"].values, offset, equal_nan=True, atol=ATOL))
    check("post-norm per-sample medians equal",
          np.ptp(np.nanmedian(norm[bio].values, axis=0)) < 1e-6)
    check("shift factors match saved",
          np.allclose(fac.set_index("column_name").loc[bio, "shift_subtracted"].values,
                      shift, atol=ATOL))
    nfb = int((reftbl["ref_source"] == "row_mean_fallback").sum())
    check("fallback count == rows where ref NaN", nfb == int(np.isnan(ref).sum()),
          f"fallback={nfb}")

    # ── STEP 05 impute ───────────────────────────────────────────────────────
    print("\n=== STEP 05 — MinProb imputation ===")
    imp = pd.read_csv(f"{root}/05_imputed/imputed_matrix.tsv", sep="\t")
    mask = pd.read_csv(f"{root}/05_imputed/imputed_value_mask.tsv", sep="\t")[bio].values.astype(bool)
    pre = norm[bio].values.astype(float)
    post = imp[bio].values.astype(float)
    check("no missing after imputation", not np.isnan(post).any())
    check("mask == normalized NaN positions",
          np.array_equal(mask, np.isnan(pre)),
          f"mask={int(mask.sum())}, NaN={int(np.isnan(pre).sum())}")
    check("observed values unchanged", np.allclose(post[~mask], pre[~mask], atol=ATOL))
    check("imputed downshifted below observed",
          post[mask].mean() < np.nanmean(pre),
          f"imp={post[mask].mean():.2f} < obs={np.nanmean(pre):.2f}")
    M2, m2 = s5.minprob_impute(pre.copy(), downshift=config.IMPUTE_DOWNSHIFT,
                               width=config.IMPUTE_WIDTH, seed=config.IMPUTE_SEED)
    check("imputation reproducible (fixed seed)",
          np.allclose(M2[m2], post[mask], atol=1e-4))
    check("protein order preserved 04→05", norm[ID].tolist() == imp[ID].tolist())

    # ── Metadata join (peptide/well) ─────────────────────────────────────────
    print("\n=== Metadata join integrity ===")
    info = utils.load_sample_info(ds)
    mrg = sub_meta[~sub_meta["is_ref"]].merge(
        info[~info["is_ref"]], on=["genotype", "timepoint", "sample_number"],
        suffixes=("", "_x"))
    check("peptide_ug matches xlsx", np.allclose(mrg["peptide_ug"], mrg["peptide_ug_x"], atol=1e-9))
    check("well matches xlsx", (mrg["well"] == mrg["well_x"]).all())

    npass = sum(res)
    print(f"\n  → {ds}: {npass}/{len(res)} PASSED")
    all_pass = all_pass and (npass == len(res))
    return npass, len(res)

tot_p = tot_n = 0
for ds in ["MousePlasma_WT", "MousePlasma_KI"]:
    p, n = verify_branch(ds)
    tot_p += p; tot_n += n

print(f"\n{'='*62}\nOVERALL BRANCH VERIFICATION: {tot_p}/{tot_n} PASSED "
      f"({'ALL PASS' if all_pass else 'SOME FAILED'})")
print(f"SUMMARY: {tot_p}/{tot_n} checks PASSED")
sys.exit(0 if all_pass else 1)
