"""Analytical-layer data-integrity verification.

Verifies that the outputs of DEP (step06), the pairwise module (step10), and
trend/time-series (step09) derive from the correct data and are internally
consistent, plus overall output completeness.  Independent recompute where
possible.  Prints PASS/FAIL per check; exits non-zero on any failure.
"""
import sys, os
import numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
sys.path.insert(0, "scripts")
import config, utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ID = config.ID_COL
MAIN = "output_data/MousePlasma"
res = []
def check(name, cond, detail=""):
    res.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

# ── DEP (step06, main) ───────────────────────────────────────────────────────
print("=== DEP (step06) integrity ===")
summ = pd.read_csv(f"{MAIN}/06_dep_limma/dep_master_summary.tsv", sep="\t")
fam = summ["family"].value_counts().to_dict()
check("20 contrasts with family counts 4/12/3/1",
      len(summ) == 20 and fam.get("genotype_at_time") == 4 and
      fam.get("time_course") == 12 and fam.get("interaction") == 3 and
      fam.get("global") == 1, fam)

norm_ids = pd.read_csv(f"{MAIN}/05_imputed/imputed_matrix.tsv", sep="\t")[ID].tolist()
n_feat = len(norm_ids)
sig_col, pthr, fcthr = config.SIG_PVALUE_COL, config.PVALUE_THRESHOLD, config.LOG2FC_THRESHOLD
all_calls_ok = all_sig_subset = all_ids_ok = True
for comp in config.COMPARISONS:
    r = pd.read_csv(f"{MAIN}/06_dep_limma/dep_results_{comp['name']}.tsv", sep="\t")
    expected_sig = (r[sig_col] < pthr) & (r["logFC"].abs() > fcthr)
    if not np.array_equal(expected_sig.values, r["Significant"].values):
        all_calls_ok = False
    if r[ID].tolist() != norm_ids:
        all_ids_ok = False
    sig_path = f"{MAIN}/06_dep_limma/dep_significant_{comp['name']}.tsv"
    if os.path.exists(sig_path):
        sig = pd.read_csv(sig_path, sep="\t")
        if set(sig[ID]) != set(r.loc[r["Significant"], ID]):
            all_sig_subset = False
    elif expected_sig.sum() > 0:
        all_sig_subset = False
check("Significant == (P.Value<0.05 & |log2FC|>1) for all 20 contrasts", all_calls_ok)
check("dep_significant IDs == Significant rows of dep_results (all contrasts)", all_sig_subset)
check("each contrast tests the full filtered protein set (no drops/dupes)",
      all_ids_ok, f"n_features={n_feat}")

# Direction integrity: logFC sign/magnitude tracks group-mean difference.
r0 = pd.read_csv(f"{MAIN}/06_dep_limma/dep_results_KI_vs_WT_8W.tsv", sep="\t")
if "mean_KI_8W" in r0.columns and "mean_WT_8W" in r0.columns:
    diff = r0["mean_KI_8W"].values - r0["mean_WT_8W"].values
    rho = np.corrcoef(r0["logFC"].values, diff)[0, 1]
    check("logFC correlates with mean_case−mean_ctrl (KI_vs_WT_8W)", rho > 0.9,
          f"corr={rho:.3f}")
else:
    check("group-mean columns present for direction check", False)

with open(f"{MAIN}/06_dep_limma/dep_count_summary.txt") as f:
    dep_txt = f.read()
check("limma design includes peptide_ug covariate", "peptide_ug" in dep_txt)

# ── Pairwise (step10, main) ──────────────────────────────────────────────────
print("\n=== Pairwise (step10) integrity ===")
psum = pd.read_csv(f"{MAIN}/10_pairwise/pairwise_separation_summary.tsv", sep="\t")
pairs = [c for c in config.COMPARISONS if c["family"] in ("genotype_at_time", "time_course")]
check("16 pairwise comparisons (4 genotype_at_time + 12 time_course)",
      len(psum) == 16 and len(pairs) == 16, f"summary={len(psum)}")

matrix = pd.read_csv(f"{MAIN}/05_imputed/imputed_matrix.tsv", sep="\t")
meta = pd.read_csv(f"{MAIN}/01_raw_summary/sample_metadata.tsv", sep="\t")
bio_meta = meta[~meta["is_ref"]]
n_ok = sil_ok = dep_ok = grp_ok = True
for comp in pairs:
    name, case_g, ctrl_g = comp["name"], comp["case_grp"], comp["ctrl_grp"]
    order = [ctrl_g, case_g]
    sub = bio_meta[bio_meta["group"].isin(order)]
    cols = sub["column_name"].tolist()
    row = psum[psum["pair"] == name].iloc[0]
    # n_samples correct
    if int(row["n_samples"]) != len(cols):
        n_ok = False
    # pca_scores rows + group membership
    pca = pd.read_csv(f"{MAIN}/10_pairwise/{name}/pca_scores.tsv", sep="\t")
    if len(pca) != len(cols) or set(pca["group"]) != set(order):
        grp_ok = False
    # silhouette recompute (PCA space, same ops as step10)
    X = matrix[cols].values.T.astype(float)
    Xs = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=min(10, Xs.shape[0] - 1),
              random_state=config.RANDOM_SEED).fit_transform(Xs)
    labels = np.array([0 if g == ctrl_g else 1 for g in sub["group"]])
    sil = silhouette_score(pcs, labels)
    if abs(sil - float(row["silhouette"])) > 1e-3:
        sil_ok = False
    # n_DEP matches the contrast's significant count
    sig_path = f"{MAIN}/06_dep_limma/dep_significant_{name}.tsv"
    n_sig = len(pd.read_csv(sig_path, sep="\t")) if os.path.exists(sig_path) else 0
    if int(row["n_DEP"]) != n_sig:
        dep_ok = False
check("each pair uses exactly the 2 groups' samples (n_samples)", n_ok)
check("pca_scores rows + group membership match the 2 groups", grp_ok)
check("silhouette recomputed from matrix matches summary (all 16)", sil_ok)
check("pairwise n_DEP == that contrast's dep_significant count", dep_ok)

# ── Trend / time-series (step09, branches) ───────────────────────────────────
print("\n=== Trend / time-series (step09) integrity ===")
for ds in ["MousePlasma_WT", "MousePlasma_KI"]:
    root = config.DATASETS[ds]["output_root"]
    td = f"{root}/09_trend_timeseries"
    tr = pd.read_csv(f"{td}/trend_results.tsv", sep="\t")
    hm = pd.read_csv(f"{td}/trend_heatmap_features.tsv", sep="\t")
    imp_ids = pd.read_csv(f"{root}/05_imputed/imputed_matrix.tsv", sep="\t")[ID].tolist()
    check(f"[{ds}] trend_results covers full branch protein set",
          len(tr) == len(imp_ids) and set(tr[ID]) == set(imp_ids), f"n={len(tr)}")
    # heatmap features all significant + direction matches slope sign + rho-ordered
    sig_ok = (hm["week_P"] < config.PVALUE_THRESHOLD).all()
    dir_ok = (((hm["direction"] == "Increasing") == (hm["week_slope"] > 0)) |
              ((hm["direction"] == "Decreasing") == (hm["week_slope"] < 0))).all()
    rho_sorted = hm["spearman_rho"].is_monotonic_increasing
    check(f"[{ds}] heatmap features: all week_P<0.05, dir=sign(slope), ρ-ascending",
          bool(sig_ok and dir_ok and rho_sorted),
          f"sig={sig_ok}, dir={dir_ok}, rho_sorted={rho_sorted}")
    # direction counts match summary
    n_inc = int((tr["direction"] == "Increasing").sum())
    n_dec = int((tr["direction"] == "Decreasing").sum())
    with open(f"{td}/trend_summary.txt") as f:
        st = f.read()
    check(f"[{ds}] inc/dec counts match trend_summary.txt",
          f"Increasing: {n_inc}" in st and f"Decreasing: {n_dec}" in st,
          f"inc={n_inc}, dec={n_dec}")
    # profile clusters: ID set == branch set, cluster in 1..k
    clu = pd.read_csv(f"{td}/profile_cluster_assignments.tsv", sep="\t")
    check(f"[{ds}] profile clusters: IDs == branch set, cluster∈1..{config.TREND_N_CLUSTERS}",
          set(clu[ID]) == set(imp_ids) and clu["cluster"].between(1, config.TREND_N_CLUSTERS).all())

# ── Mouse-brain protein annotation (step01b) + brain-filtered re-analysis (step11) ─
print("\n=== Mouse-brain protein annotation (step01b) integrity ===")
raw_path = f"{MAIN}/01_raw_summary/raw_intensity_matrix.tsv"
ann_path = f"{MAIN}/01_raw_summary/mouse_brain_annotation.tsv"
ann_cols = ["is_mouse_brain_protein", "mouse_brain_regional_category", "is_regionally_elevated"]
if os.path.exists(raw_path) and os.path.exists(ann_path):
    raw = pd.read_csv(raw_path, sep="\t")
    ann = pd.read_csv(ann_path, sep="\t")
    check("raw_intensity_matrix.tsv carries all 3 annotation columns",
          all(c in raw.columns for c in ann_cols), str(raw.columns[-3:].tolist()))
    check("raw_intensity_matrix.tsv row count/IDs unchanged by annotation (disk-only augmentation)",
          len(raw) == 9769 and raw[ID].is_unique, f"n={len(raw)}")
    check("is_regionally_elevated implies is_mouse_brain_protein (annotation table)",
          (ann.loc[ann["is_regionally_elevated"], "is_mouse_brain_protein"] == True).all())
    check("is_regionally_elevated implies is_mouse_brain_protein (raw matrix copy)",
          (raw.loc[raw["is_regionally_elevated"] == True, "is_mouse_brain_protein"] == True).all())
    check("mouse_brain_regional_category ⊆ config.BRAIN_ELEVATED_CATEGORIES ∪ {Low region specificity, ''}",
          set(ann.loc[ann["is_mouse_brain_protein"], "mouse_brain_regional_category"].unique())
          <= set(config.BRAIN_ELEVATED_CATEGORIES) | {"Low region specificity"})
    check("every accession has a cached UniProt gene-name lookup (no silent skips)",
          os.path.exists(config.UNIPROT_GENE_NAME_CACHE) and
          set(ann[ID]) <= set(pd.read_csv(config.UNIPROT_GENE_NAME_CACHE, sep="\t")["accession"]))

    print("\n=== Brain-protein-filtered re-analysis (step11) integrity ===")
    step11_dir = f"{MAIN}/11_brain_filtered"
    filt = pd.read_csv(f"{MAIN}/02_filtered/filtered_matrix.tsv", sep="\t")
    expected_brain_ids = set(ann.loc[ann["is_mouse_brain_protein"], ID]) & set(filt[ID])
    bp = pd.read_csv(f"{step11_dir}/brain_protein_matrix.tsv", sep="\t")
    check("step11 subset == is_mouse_brain_protein ∩ filtered-protein set (no drift)",
          set(bp[ID]) == expected_brain_ids, f"step11 n={len(bp)}, expected={len(expected_brain_ids)}")
    pca = pd.read_csv(f"{step11_dir}/pca_scores.tsv", sep="\t")
    check("step11 PCA covers all 80 biological samples",
          len(pca) == 80 and set(pca["group"]) == set(config.PLASMA_GROUP_ORDER))
    summ = pd.read_csv(f"{step11_dir}/brain_filtered_separation_summary.tsv", sep="\t")
    check("step11 separation summary covers genotype/timepoint/group axes",
          set(summ["axis"]) == {"genotype", "timepoint", "group"})

    print("\n=== Regionally-elevated-only re-analysis (step12) integrity ===")
    step12_dir = f"{MAIN}/12_regionally_elevated"
    db = pd.read_csv(config.MOUSE_BRAIN_DB, sep="\t", low_memory=False)
    db_elevated_mask = db[config.BRAIN_DB_REGIONAL_COL].isin(config.BRAIN_ELEVATED_CATEGORIES)
    db_elevated_genes = set(db.loc[db_elevated_mask, config.BRAIN_DB_GENE_COL].str.upper())
    check("DB regionally-elevated gene count == 763 (333+266+164, confirmed 2026-07-13)",
          len(db_elevated_genes) == 763, f"n={len(db_elevated_genes)}")

    cov = pd.read_csv(f"{step12_dir}/regionally_elevated_coverage_report.tsv", sep="\t")
    check("coverage report: DB stage == 763 genes",
          int(cov.loc[cov["stage"].str.startswith("DB"), "n_genes"].iloc[0]) == 763)
    expected_elevated_ids = set(ann.loc[ann["is_regionally_elevated"], ID]) & set(filt[ID])
    re = pd.read_csv(f"{step12_dir}/regionally_elevated_matrix.tsv", sep="\t")
    check("step12 subset == is_regionally_elevated ∩ filtered-protein set (no drift)",
          set(re[ID]) == expected_elevated_ids, f"step12 n={len(re)}, expected={len(expected_elevated_ids)}")
    check("step12 subset ⊆ step11 subset (elevated is a subset of all brain proteins)",
          set(re[ID]) <= set(bp[ID]))
    # pandas auto-infers "True"/"False" text columns as bool on read_csv, so
    # accept either representation (the file itself stores the literal text
    # "True"/"False", verified separately by inspecting the raw bytes).
    mbp_all_true = (re["Mouse_Brain_Protein"] == "True").all() or \
                  (re["Mouse_Brain_Protein"] == True).all()
    check("regionally_elevated_matrix.tsv: Mouse_Brain_Protein all 'True', Regional_Category "
          "∈ BRAIN_ELEVATED_CATEGORIES (last 2 columns)",
          mbp_all_true and
          set(re["Regional_Category"].unique()) <= set(config.BRAIN_ELEVATED_CATEGORIES))
    pca12 = pd.read_csv(f"{step12_dir}/pca_scores.tsv", sep="\t")
    check("step12 PCA covers all 80 biological samples",
          len(pca12) == 80 and set(pca12["group"]) == set(config.PLASMA_GROUP_ORDER))
else:
    check("step01b/step11 outputs present (raw_intensity_matrix + mouse_brain_annotation)", False)

# ── Completeness / no-corruption ─────────────────────────────────────────────
print("\n=== Completeness / no-corruption ===")
for ds in ["MousePlasma", "MousePlasma_WT", "MousePlasma_KI"]:
    root = config.DATASETS[ds]["output_root"]
    m = pd.read_csv(f"{root}/05_imputed/imputed_matrix.tsv", sep="\t")
    bio = [c for c in m.columns if c not in config.PLASMA_META_COLS]
    check(f"[{ds}] final imputed matrix has no NaN", not m[bio].isna().any().any())
stray = []
for dp, dn, fn in os.walk("output_data"):
    for d in dn:
        if d and d[-1].isdigit() and d[-2:-1] == " ":
            stray.append(os.path.join(dp, d))
check("no sync-artifact stray directories", len(stray) == 0, str(stray[:3]))
check("pipeline config snapshot exists",
      os.path.exists("output_data/_shared/logs/pipeline_config_snapshot.json"))

print("\n" + "=" * 60)
print(f"SUMMARY: {sum(res)}/{len(res)} analytical-layer checks PASSED")
sys.exit(0 if all(res) else 1)
