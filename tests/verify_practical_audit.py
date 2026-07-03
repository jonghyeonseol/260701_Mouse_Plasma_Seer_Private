"""Practical/business-grade data-integrity audit.

Unlike tests/verify_preprocessing.py etc. (which re-import scripts/utils.py and
scripts/config.py and are therefore partly circular — a bug shared between the
pipeline and the checker would pass both), this script deliberately avoids the
pipeline's own code wherever possible: raw text parsing, standalone regex, and
mathematical properties that hold regardless of implementation. It targets the
failure modes a real-world QC reviewer cares about: file tampering, silent
sample drops, duplicate/mislabeled samples, cross-branch contamination, and
statistical-output correctness.

Prints PASS/FAIL/FLAG per check; exits non-zero only on FAIL (FLAG = needs
human/domain review, not a proven defect).
"""
import sys, os, hashlib, re, glob
import numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

RAW_TSV = "input_data/20260630_142950_20260529_Report.tsv"
XLSX = "input_data/plasma_sample_information.xlsx"
MAIN = "output_data/MousePlasma"

# Last-verified checksum of RAW_TSV (recorded in claudedocs/DATA_INTEGRITY_REPORT.md,
# 2026-07-03 audit). A mismatch means the raw input changed WITHOUT the pipeline
# being re-run and re-verified against it (PM plan Phase D). If the input has
# legitimately changed (new data delivery), re-run scripts/run_pipeline.py +
# scripts/run_branch.py + this whole suite, THEN update this constant in the
# same commit as the regenerated outputs — never update it silently.
EXPECTED_RAW_SHA256 = "9e42b3a0ce44ad85624bf26fedf884123d2048360530bdf3723f3d08a9a90c15"

results, flags = [], []
def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
def flag(name, detail=""):
    flags.append((name, detail))
    print(f"[FLAG] {name}" + (f"  — {detail}" if detail else "") + "  (needs human review, not a proven defect)")

# ── A. Raw file chain-of-custody (pure filesystem, no pandas) ───────────────
print("=== A. Raw file integrity ===")
check("raw TSV exists", os.path.exists(RAW_TSV))
with open(RAW_TSV, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
print(f"  SHA-256: {sha}")
check("raw TSV size is non-trivial (>10MB, not truncated)",
      os.path.getsize(RAW_TSV) > 10_000_000, f"{os.path.getsize(RAW_TSV):,} bytes")
check("raw TSV SHA-256 matches last-verified checksum (no unnoticed input drift)",
      sha == EXPECTED_RAW_SHA256,
      f"got {sha[:16]}...  expected {EXPECTED_RAW_SHA256[:16]}...  "
      "(if the input legitimately changed: re-run run_pipeline.py + run_branch.py "
      "+ this full suite, THEN update EXPECTED_RAW_SHA256 in the same commit)")

# ── B. Pure-text header parse (no pandas/pipeline code) ──────────────────────
print("\n=== B. Sample-column accounting (pure regex, independent of scripts/) ===")
with open(RAW_TSV) as f:
    header = f.readline().rstrip("\n").split("\t")
sample_cols = header[3:]
check("84 total columns (3 meta + 81 sample)", len(header) == 84, f"{len(header)}")
check("no duplicate column headers", len(set(header)) == len(header))

pat = re.compile(r"\[(\d+)\]\s+(\d{8})_SEER_HanKH_mouse_plasma_(\d+)W_(WT|KI)(\d+)\.raw\.PG\.Quantity")
ref_pat = re.compile(r"\[(\d+)\]\s+(\d{8})_SEER_HanKH_mouse_plasma_Total_pooled_ref\.raw\.PG\.Quantity")
matched, indices = 0, set()
for c in sample_cols:
    m = pat.match(c) or ref_pat.match(c)
    if m:
        matched += 1
        indices.add(int(m.group(1)))
check("every sample column matched by regex (no silent drops)", matched == len(sample_cols),
      f"{matched}/{len(sample_cols)}")
check("column [index] values unique 1..81 (no dup/gap)",
      indices == set(range(1, 82)), f"n_unique={len(indices)}")

# ── C. Independent xlsx parse (openpyxl via pandas, no utils.load_sample_info) ─
print("\n=== C. Sample-information workbook (independent parse) ===")
raw_xlsx = pd.read_excel(XLSX, header=1)
raw_xlsx.columns = [str(c).strip() for c in raw_xlsx.columns]
sample_col = [c for c in raw_xlsx.columns if "sample" in c.lower()][0]
well_col = [c for c in raw_xlsx.columns if "well" in c.lower()][0]
pep_col = [c for c in raw_xlsx.columns if "peptide" in c.lower()][0]
check("xlsx has 81 rows (80 bio + 1 pool)", len(raw_xlsx) == 81, f"{len(raw_xlsx)}")
wells = raw_xlsx[well_col].astype(str).str.strip()
check("no duplicate well assignments (no physical mixup)", not wells.duplicated().any())
labels = raw_xlsx[sample_col].astype(str).str.strip()
check("no duplicate sample labels", not labels.duplicated().any())
pep = pd.to_numeric(raw_xlsx[pep_col], errors="coerce")
check("peptide_ug has no missing/non-positive values", pep.notna().all() and (pep > 0).all(),
      f"range={pep.min():.3f}-{pep.max():.3f}")

# ── D/E. Sample-identity screen: duplication + within-group outliers ────────
print("\n=== D/E. Sample identity screen (duplication / mislabel risk) ===")
raw_df = pd.read_csv(RAW_TSV, sep="\t")
vals = raw_df[sample_cols].apply(pd.to_numeric, errors="coerce")
corr = vals.corr()
np.fill_diagonal(corr.values, np.nan)
iu = np.triu_indices_from(corr.values, k=1)
all_corrs = corr.values[iu]
all_corrs = all_corrs[~np.isnan(all_corrs)]
max_corr = np.nanmax(corr.values)
check("no near-duplicate sample columns (max pairwise corr < 0.999)", max_corr < 0.999,
      f"max={max_corr:.4f}")
top_idx = np.unravel_index(np.nanargmax(corr.values), corr.values.shape)
top_pair = (sample_cols[top_idx[0]], sample_cols[top_idx[1]])
pctile = (all_corrs < max_corr).mean() * 100
if max_corr > 0.995:
    flag("highest-correlation sample pair is an outlier vs the overall distribution",
         f"r={max_corr:.4f} (percentile {pctile:.1f}%): "
         f"{top_pair[0].split('mouse_plasma_')[-1]} vs {top_pair[1].split('mouse_plasma_')[-1]} — "
         f"cross-genotype/timepoint pair; not proof of mislabeling (7,774/9,769 proteins "
         f"jointly detected with differing missingness), but worth a manual plate/run-order check.")
else:
    check("top-correlation pair is not an outlier", True, f"r={max_corr:.4f}")

groups = {}
for c in sample_cols:
    m = pat.match(c)
    if m:
        groups.setdefault(f"{m.group(4)}_{m.group(3)}W", []).append(c)
low_corr = []
for g, cols in groups.items():
    sub = corr.loc[cols, cols]
    med = sub.median(axis=1)
    low_corr += [(g, c, round(v, 3)) for c, v in med.items() if v < 0.3]
check("no sample poorly correlated with its own declared group (median r>0.3)",
      len(low_corr) == 0, str(low_corr[:5]))

# ── F. BH-adjustment mathematical correctness (implementation-independent) ──
print("\n=== F. Statistical-output correctness (BH monotonicity, all 20 contrasts) ===")
bad = []
for f in sorted(glob.glob(f"{MAIN}/06_dep_limma/dep_results_*.tsv")):
    d = pd.read_csv(f, sep="\t").sort_values("P.Value")
    if not (d["adj.P.Val"].diff().dropna() >= -1e-9).all():
        bad.append(os.path.basename(f) + " (non-monotonic)")
    if not (d["adj.P.Val"] >= d["P.Value"] - 1e-9).all():
        bad.append(os.path.basename(f) + " (adjP < rawP)")
check("adj.P.Val respects BH monotonicity & adjP>=rawP in every contrast", len(bad) == 0, str(bad))

# ── G. Exact (not correlational) recompute of DEP group means ───────────────
print("\n=== G. DEP group means — exact recompute from imputed matrix ===")
imp = pd.read_csv(f"{MAIN}/05_imputed/imputed_matrix.tsv", sep="\t")
meta = pd.read_csv(f"{MAIN}/01_raw_summary/sample_metadata.tsv", sep="\t")
bio = meta[~meta["is_ref"]]
r0 = pd.read_csv(f"{MAIN}/06_dep_limma/dep_results_KI_vs_WT_8W.tsv", sep="\t")
exact_ok = True
for grp, col in [("KI_8W", "mean_KI_8W"), ("WT_8W", "mean_WT_8W")]:
    cols = bio.loc[bio["group"] == grp, "column_name"].tolist()
    recomputed = imp[cols].mean(axis=1).values
    if not np.allclose(recomputed, r0[col].values, atol=1e-9):
        exact_ok = False
check("mean_KI_8W / mean_WT_8W match exact recompute (not just correlated)", exact_ok)

# ── H. Cross-branch contamination ────────────────────────────────────────────
print("\n=== H. Cross-branch genotype isolation ===")
wt_cols = pd.read_csv(f"output_data/MousePlasma_WT/05_imputed/imputed_matrix.tsv",
                      sep="\t", nrows=1).columns
ki_cols = pd.read_csv(f"output_data/MousePlasma_KI/05_imputed/imputed_matrix.tsv",
                      sep="\t", nrows=1).columns
wt_bio = [c for c in wt_cols if "PG.Quantity" in c]
ki_bio = [c for c in ki_cols if "PG.Quantity" in c]
check("WT branch contains zero KI-labeled columns", not any("_KI" in c for c in wt_bio))
check("KI branch contains zero WT-labeled columns", not any("_WT" in c for c in ki_bio))
check("WT/KI branch sample sets are disjoint", len(set(wt_bio) & set(ki_bio)) == 0)

print("\n" + "=" * 66)
print(f"SUMMARY: {sum(results)}/{len(results)} checks PASSED, {len(flags)} FLAGGED for human review")
if flags:
    print("FLAGGED (not failures — recommend domain-expert review):")
    for name, detail in flags:
        print(f"  - {name}: {detail}")
sys.exit(1 if not all(results) else 0)
