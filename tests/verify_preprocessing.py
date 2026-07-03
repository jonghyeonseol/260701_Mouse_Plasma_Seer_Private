"""Independent verification of the preprocessing pipeline (steps 01-05).

Re-derives each transformation directly from the raw TSV + xlsx and compares to
the pipeline's saved matrices.  Prints PASS/FAIL per check.
"""
import sys, os
import numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
sys.path.insert(0, "scripts")
import config, utils

ID = config.ID_COL
ROOT = "output_data/MousePlasma"
RTOL, ATOL = 1e-6, 1e-6
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

# ── Ground truth: raw load via the same IO, plus a fully independent raw read ──
raw_df, meta = utils.load_raw("MousePlasma")
all_cols = meta["column_name"].tolist()
raw_num = utils.to_numeric_intensity(raw_df, all_cols)
bio_cols = meta.loc[~meta["is_ref"], "column_name"].tolist()
ref_col = meta.loc[meta["is_ref"], "column_name"].iloc[0]
groups = config.PLASMA_GROUP_ORDER

# Independent raw read (no project code) for cross-check of numeric coercion.
indep = pd.read_csv(config.PLASMA_INPUT, sep="\t")
indep = indep[[c for c in indep.columns if not str(c).startswith("Unnamed:")]]

print("\n=== STEP 01 — raw parsing ===")
check("81 sample columns parsed (80 bio + 1 ref)",
      len(bio_cols) == 80 and meta["is_ref"].sum() == 1,
      f"bio={len(bio_cols)}, ref={int(meta['is_ref'].sum())}")
gc = meta.loc[~meta["is_ref"], "group"].value_counts().reindex(groups)
check("group sizes match design (10,11,9,10,10,12,8,10)",
      gc.tolist() == [10,11,9,10,10,12,8,10], dict(gc))
# zeros/negatives coerced to NaN
raw_vals = raw_num[all_cols].values
check("no non-positive intensities remain (zeros→NaN)",
      np.nanmin(raw_vals) > 0, f"min={np.nanmin(raw_vals):.3f}")
# independent numeric equality on a column
c0 = bio_cols[0]
a = pd.to_numeric(indep[c0], errors="coerce").mask(lambda s: s <= 0)
b = raw_num[c0]
check("numeric coercion matches independent read (col 1)",
      np.allclose(a.values, b.values, equal_nan=True))
# saved raw matrix equals in-memory
saved_raw = pd.read_csv(f"{ROOT}/01_raw_summary/raw_intensity_matrix.tsv", sep="\t")
check("saved raw_intensity_matrix == recomputed",
      np.allclose(saved_raw[all_cols].values, raw_num[all_cols].values, equal_nan=True),
      f"shape={saved_raw.shape}")

print("\n=== STEP 02 — valid-fraction filter (≥50% in ALL 8 groups) ===")
filt = pd.read_csv(f"{ROOT}/02_filtered/filtered_matrix.tsv", sep="\t")
# independent keep mask
vals = raw_num[bio_cols].values
keep = np.ones(vals.shape[0], dtype=bool)
for g in groups:
    gcols = meta.loc[meta["group"] == g, "column_name"].tolist()
    idx = [bio_cols.index(c) for c in gcols]
    frac = np.sum(~np.isnan(vals[:, idx]), axis=1) / len(idx)
    keep &= (frac >= 0.50)
kept_ids_indep = set(raw_num.loc[keep, ID])
kept_ids_saved = set(filt[ID])
check("filtered protein set == independent recompute",
      kept_ids_indep == kept_ids_saved,
      f"indep={len(kept_ids_indep)}, saved={len(kept_ids_saved)}")
check("filter count 9769 → 7899",
      raw_num.shape[0] == 9769 and filt.shape[0] == 7899,
      f"{raw_num.shape[0]}→{filt.shape[0]}")
# every kept protein truly ≥50% in every group; every removed fails ≥1 group
removed = raw_num.loc[~keep]
def min_group_frac(row_idx):
    fr = []
    for g in groups:
        gcols = meta.loc[meta["group"] == g, "column_name"].tolist()
        idx = [bio_cols.index(c) for c in gcols]
        fr.append(np.sum(~np.isnan(vals[row_idx, idx])) / len(idx))
    return min(fr)
sample_kept = np.where(keep)[0][:200]
sample_rem = np.where(~keep)[0][:200]
check("all kept proteins ≥50% valid in every group",
      all(min_group_frac(i) >= 0.50 for i in sample_kept))
check("all removed proteins <50% in ≥1 group",
      all(min_group_frac(i) < 0.50 for i in sample_rem))

print("\n=== STEP 03 — log2 transform ===")
log2 = pd.read_csv(f"{ROOT}/03_log2_transformed/log2_matrix.tsv", sep="\t")
# log2_matrix should equal log2(filtered raw) for bio+ref cols
recon = np.log2(filt[bio_cols + [ref_col]].values.astype(float))
check("log2_matrix == log2(filtered intensities)",
      np.allclose(log2[bio_cols + [ref_col]].values, recon, equal_nan=True, atol=ATOL))
check("log2 finite where raw present (no -inf from log2(0))",
      np.isfinite(log2[bio_cols].values[~np.isnan(filt[bio_cols].values)]).all())

print("\n=== STEP 04 — reference-ratio + median normalization ===")
norm = pd.read_csv(f"{ROOT}/04_normalized/normalized_matrix.tsv", sep="\t")
factors = pd.read_csv(f"{ROOT}/04_normalized/normalization_factors.tsv", sep="\t")
reftbl = pd.read_csv(f"{ROOT}/04_normalized/reference_ratio_table.tsv", sep="\t")
# ref column dropped
check("pooled-ref column dropped from normalized matrix",
      ref_col not in norm.columns and norm.shape[1] == 3 + 80, f"cols={norm.shape[1]}")
# independent reconstruction from log2 (bio + ref)
L = log2[bio_cols].values.astype(float)
ref = log2[ref_col].values.astype(float)
row_mean = np.nanmean(L, axis=1)
offset = np.where(np.isnan(ref), row_mean, ref)
offset = np.where(np.isnan(offset), 0.0, offset)
ratio = L - offset[:, None]
per_sample_med = np.nanmedian(ratio, axis=0)
global_med = np.nanmedian(per_sample_med)
shift = per_sample_med - global_med
recon_norm = ratio - shift[None, :]
check("normalized_matrix == independent (ref-ratio then median)",
      np.allclose(norm[bio_cols].values, recon_norm, equal_nan=True, atol=1e-5))
# offset table consistency
check("reference_ratio_table.offset matches recompute",
      np.allclose(reftbl["offset_subtracted"].values, offset, equal_nan=True, atol=ATOL))
n_fallback = int((reftbl["ref_source"] == "row_mean_fallback").sum())
check("ref_source fallback count == rows where ref is NaN",
      n_fallback == int(np.isnan(ref).sum()), f"fallback={n_fallback}")
# post-normalization per-sample medians equal
post_med = np.nanmedian(norm[bio_cols].values, axis=0)
check("post-norm per-sample medians all ≈ equal",
      np.ptp(post_med) < 1e-6, f"range={np.ptp(post_med):.2e}")
# median shift factors match saved
check("normalization_factors.shift_subtracted matches recompute",
      np.allclose(factors.set_index("column_name").loc[bio_cols, "shift_subtracted"].values,
                  shift, atol=ATOL))

print("\n=== STEP 05 — MinProb imputation ===")
imp = pd.read_csv(f"{ROOT}/05_imputed/imputed_matrix.tsv", sep="\t")
mask = pd.read_csv(f"{ROOT}/05_imputed/imputed_value_mask.tsv", sep="\t")
M = imp[bio_cols].values.astype(float)
Msk = mask[bio_cols].values.astype(bool)
pre = norm[bio_cols].values.astype(float)
check("imputed matrix has NO missing values", not np.isnan(M).any())
check("mask == positions missing in normalized matrix",
      np.array_equal(Msk, np.isnan(pre)),
      f"mask sum={int(Msk.sum())}, norm NaN={int(np.isnan(pre).sum())}")
check("observed (non-imputed) values unchanged vs normalized",
      np.allclose(M[~Msk], pre[~Msk], atol=ATOL))
# MinProb property: imputed values are downshifted below observed
obs_mean = np.nanmean(pre)
imp_mean = M[Msk].mean()
check("imputed values downshifted below observed (MinProb)",
      imp_mean < obs_mean, f"imp_mean={imp_mean:.2f} < obs_mean={obs_mean:.2f}")
# per-protein: for a protein with ≥2 obs and some missing, imputed mean ≈ mu-1.8*sd
rng_ok = []
for i in range(M.shape[0]):
    o = pre[i][~Msk[i]]; miss = np.where(Msk[i])[0]
    if o.size >= 2 and miss.size >= 3:
        mu, sd = o.mean(), o.std(ddof=1)
        target = mu - config.IMPUTE_DOWNSHIFT * sd
        rng_ok.append(abs(M[i, miss].mean() - target) <= 1.2 * max(sd, 1e-6))
    if len(rng_ok) >= 300: break
check("per-protein imputed mean ≈ mu − 1.8·sd (within tolerance)",
      np.mean(rng_ok) > 0.9, f"{np.mean(rng_ok)*100:.1f}% of {len(rng_ok)} sampled proteins")
# reproducibility: re-run imputer with same seed reproduces values
import step05_impute as s5
M2, msk2 = s5.minprob_impute(pre.copy(), downshift=config.IMPUTE_DOWNSHIFT,
                             width=config.IMPUTE_WIDTH, seed=config.IMPUTE_SEED)
check("imputation reproducible with fixed seed",
      np.allclose(M2[msk2], M[Msk], atol=1e-4))

print("\n" + "=" * 60)
n_pass = sum(1 for _, ok, _ in results if ok)
print(f"SUMMARY: {n_pass}/{len(results)} checks PASSED")
fails = [n for n, ok, _ in results if not ok]
if fails:
    print("FAILED:", fails)
sys.exit(1 if fails else 0)
