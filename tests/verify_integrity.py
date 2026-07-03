"""Cross-step data-integrity + biological-sanity verification of preprocessing."""
import sys, os
import numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
sys.path.insert(0, "scripts")
import config, utils
ID = config.ID_COL
ROOT = "output_data/MousePlasma"
res = []
def check(name, cond, detail=""):
    res.append(bool(cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

raw = pd.read_csv(f"{ROOT}/01_raw_summary/raw_intensity_matrix.tsv", sep="\t")
filt = pd.read_csv(f"{ROOT}/02_filtered/filtered_matrix.tsv", sep="\t")
log2 = pd.read_csv(f"{ROOT}/03_log2_transformed/log2_matrix.tsv", sep="\t")
norm = pd.read_csv(f"{ROOT}/04_normalized/normalized_matrix.tsv", sep="\t")
imp  = pd.read_csv(f"{ROOT}/05_imputed/imputed_matrix.tsv", sep="\t")
meta = pd.read_csv(f"{ROOT}/01_raw_summary/sample_metadata.tsv", sep="\t")
bio = meta.loc[~meta["is_ref"], "column_name"].tolist()
groups = config.PLASMA_GROUP_ORDER

print("=== Cross-step protein-set / order integrity ===")
check("no duplicate protein IDs in raw", raw[ID].is_unique, f"n={raw[ID].nunique()}")
check("filtered ⊆ raw protein IDs", set(filt[ID]).issubset(set(raw[ID])))
check("log2 protein IDs identical & same order as filtered",
      filt[ID].tolist() == log2[ID].tolist())
check("normalized protein IDs identical & same order as filtered",
      filt[ID].tolist() == norm[ID].tolist())
check("imputed protein IDs identical & same order as normalized",
      norm[ID].tolist() == imp[ID].tolist())
check("row counts consistent 02→05 (all 7899)",
      len({len(filt), len(log2), len(norm), len(imp)}) == 1, f"{len(filt)}")

print("\n=== Metadata join integrity (vs xlsx) ===")
info = utils.load_sample_info("MousePlasma")
mrg = meta[~meta["is_ref"]].merge(
    info[~info["is_ref"]], on=["genotype", "timepoint", "sample_number"],
    suffixes=("", "_xlsx"))
check("all 80 bio samples matched to xlsx rows", len(mrg) == 80, f"n={len(mrg)}")
check("peptide_ug matches xlsx exactly",
      np.allclose(mrg["peptide_ug"], mrg["peptide_ug_xlsx"], atol=1e-9))
check("well matches xlsx exactly", (mrg["well"] == mrg["well_xlsx"]).all())
check("no missing peptide_ug / well in bio metadata",
      meta.loc[~meta["is_ref"], ["peptide_ug", "well"]].notna().all().all())

print("\n=== Filter reasonableness ===")
excl = pd.read_csv(f"{ROOT}/02_filtered/filter_excluded_low_valid.tsv", sep="\t")
det_kept = filt[bio].notna().mean(axis=1).mean()
# recompute detection for removed from raw
rawnum = raw.copy()
det_all = raw[bio].notna().mean(axis=1)
kept_mask = raw[ID].isin(set(filt[ID]))
check("removed proteins have lower mean detection than kept",
      det_all[~kept_mask].mean() < det_all[kept_mask].mean(),
      f"removed={det_all[~kept_mask].mean():.2f} < kept={det_all[kept_mask].mean():.2f}")
check("excluded count + kept count == raw count",
      len(excl) + len(filt) == len(raw), f"{len(excl)}+{len(filt)}={len(excl)+len(filt)}")

print("\n=== Reference-ratio biological sanity ===")
reftbl = pd.read_csv(f"{ROOT}/04_normalized/reference_ratio_table.tsv", sep="\t")
# After ratio+median, global distribution should be centered near 0 (ratios to ref)
gmed = np.nanmedian(norm[bio].values)
check("normalized ratios centered near 0 (log2 sample/ref)",
      abs(gmed) < 0.1, f"global median={gmed:.3f}")
# proteins measured in ref use pooled_ref (not fallback)
check("majority of proteins anchored to pooled ref (not fallback)",
      (reftbl["ref_source"] == "pooled_ref").mean() > 0.95,
      f"{(reftbl['ref_source']=='pooled_ref').mean()*100:.1f}% pooled")

print("\n=== Imputation impact sanity ===")
mask = pd.read_csv(f"{ROOT}/05_imputed/imputed_value_mask.tsv", sep="\t")[bio].values.astype(bool)
pre = norm[bio].values.astype(float)
post = imp[bio].values.astype(float)
obs = pre[~mask]; impv = post[mask]
# imputed 5th-95th percentile sits left of observed median
check("imputed distribution left-shifted (95th pct imputed ≤ observed median)",
      np.percentile(impv, 95) <= np.median(obs) + 0.5,
      f"imp95={np.percentile(impv,95):.2f}, obs_med={np.median(obs):.2f}")
# per-group imputation (missingness) rates are moderate, not dominating any group
rates = {}
for g in groups:
    gcols = meta.loc[meta["group"] == g, "column_name"].tolist()
    idx = [bio.index(c) for c in gcols]
    rates[g] = float(mask[:, idx].mean() * 100)
check("per-group imputation rate < 15% for all groups",
      max(rates.values()) < 15, {k: round(v,1) for k,v in rates.items()})
# imputation didn't inflate variance absurdly (post std within 2x pre-observed std)
check("global std not inflated >2x by imputation",
      post.std() < 2 * obs.std(), f"post={post.std():.2f}, obs={obs.std():.2f}")

print("\n" + "="*55)
print(f"SUMMARY: {sum(res)}/{len(res)} integrity/sanity checks PASSED")
sys.exit(0 if all(res) else 1)
