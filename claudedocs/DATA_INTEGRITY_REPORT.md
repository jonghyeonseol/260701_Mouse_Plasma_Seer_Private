# Data Integrity Report — Mouse Plasma SEER Pipeline

**Date:** 2026-07-03 (practical audit extended 2026-07-04 with an input-drift check)
**Scope:** Final data-integrity verification across all three projects — `MousePlasma`
(main, 8 groups), `MousePlasma_WT`, `MousePlasma_KI` (per-genotype branches) — after the
20-contrast DEP expansion, the pairwise module (step10), the ComplexHeatmap trend heatmap,
and colormap/layout changes.

**Result: ✅ 108 / 108 automated checks PASSED, plus a practical audit — 18 / 18 PASSED
with 1 item flagged for human review (not a proven defect).** See §5.

> **Methodological note (added on practical review):** §§1–4 below re-derive each transform
> using `scripts/utils.py` / `scripts/config.py` — this is strong **regression/consistency**
> testing but is partly circular: a shared bug in those modules would pass both the pipeline
> and the checker. §5 (`tests/verify_practical_audit.py`) was added specifically to close
> this gap with checks that avoid the pipeline's own code (raw text/regex parsing, file
> checksums, correlation-based sample-identity screening, and implementation-independent
> mathematical properties). Both are necessary; neither alone is sufficient.

Verification was performed by **independent recomputation**: each transform was re-derived
directly from the raw TSV + sample-information xlsx and compared to the pipeline's saved
matrices. Outputs were regenerated cleanly from the current code (`run_pipeline.py` +
`run_branch.py`, logs error-free) immediately before verification.

| Suite | Script | Checks | Result |
|-------|--------|:------:|:------:|
| Preprocessing correctness (steps 01–05, main) | `tests/verify_preprocessing.py` | 23 | ✅ PASS |
| Cross-step integrity + metadata + sanity (main) | `tests/verify_integrity.py` | 17 | ✅ PASS |
| Branch preprocessing (WT + KI, steps 02–05) | `tests/verify_branch.py` | 44 | ✅ PASS |
| Analytical layer (DEP / pairwise / trend / completeness) | `tests/verify_analysis.py` | 24 | ✅ PASS |
| **Subtotal (regression-style, §§1–4)** | | **108** | **✅ PASS** |
| Practical audit — implementation-independent | `tests/verify_practical_audit.py` | 18 | ✅ PASS (+1 flag) |

---

## 1. Preprocessing correctness (23/23)
- **Parsing:** 81 columns → 80 biological + 1 pooled ref; group sizes match design
  (10/11/9/10/10/12/8/10); zeros/negatives coerced to NaN; numeric read matches an
  independent parse; saved raw matrix == recompute.
- **Filter (step02):** kept protein set == independent ≥50%-in-all-8-groups recompute
  (9,769 → 7,899); every kept protein ≥50% in every group, every removed fails ≥1 group.
- **log2 (step03):** `log2_matrix == log2(filtered)`; no −inf.
- **Normalize (step04):** `normalized == independent (ref-ratio then median)`; offset table
  and shift factors match; pooled-ref column dropped; post-norm per-sample medians equal
  (range 5.2e-17); ref fallback count == rows with missing ref (63).
- **Impute (step05):** no NaN remain; mask == normalized-NaN positions (32,264); observed
  values unchanged; MinProb downshift confirmed; per-protein imputed mean ≈ μ−1.8·sd (100%
  of sampled); **reproducible with fixed seed**.

## 2. Cross-step integrity + metadata (17/17)
- **No scrambling:** protein IDs unique; identical ID set & row order preserved 02→05; row
  counts consistent (7,899).
- **Metadata join:** all 80 biological samples matched to the xlsx; `peptide_ug` and `well`
  match the workbook exactly; no missing metadata.
- **Sanity:** removed proteins have lower detection than kept (0.41 vs 0.95); excluded+kept
  == raw; normalized ratios centered near 0; 99.2% anchored to pooled ref; imputed
  distribution left-shifted; per-group imputation 3.0–8.5% (<15%); variance not inflated.

## 3. Branch preprocessing (44/44 — WT 22, KI 22)
- **Subset isolation:** each branch = 40 samples of its genotype (+pooled ref); all samples
  the target genotype (no cross-contamination); 4 timepoint groups with expected sizes.
- **Re-filtering applied:** WT 8,113 / KI 8,127 features (independent recompute matches;
  differs from the main 7,899 because only 4 groups constrain the ≥50% filter).
- log2 / ref-ratio+median / MinProb all match independent recompute; order preserved;
  metadata join exact; imputation reproducible.

## 4. Analytical-layer integrity (24/24)
- **DEP (step06):** 20 contrasts with family counts 4/12/3/1; `Significant ==
  (P.Value<0.05 & |log2FC|>1)` for all; `dep_significant` == the significant rows of
  `dep_results`; every contrast tests the full 7,899-protein set; **direction integrity** —
  logFC vs (mean_KI−mean_WT) corr = 0.950; peptide covariate present in the design.
- **Pairwise (step10):** 16 comparisons (4+12); each uses exactly its 2 groups' samples;
  `pca_scores` rows & group membership correct; **silhouette recomputed from the matrix
  matches the summary for all 16**; `n_DEP` == that contrast's `dep_significant` count.
- **Trend (step09, WT+KI):** `trend_results` covers the full branch protein set; heatmap
  features are all `week_P<0.05`, direction == sign(week_slope), rows ρ-ascending; inc/dec
  counts match `trend_summary.txt` (WT 258/617, KI 138/340); profile-cluster IDs == branch
  set, clusters ∈ 1..6.
- **Completeness:** final imputed matrices (all 3 projects) contain no NaN; no sync-artifact
  stray directories; pipeline config snapshot present.

## 5. Practical audit — implementation-independent (18/18 PASS, 1 FLAG)

Added specifically to check what §§1–4 structurally cannot, because they share code with the
pipeline. This suite parses the raw TSV with plain Python (no pandas/pipeline import) and the
xlsx standalone, checks file-level chain of custody, screens for sample duplication/mislabeling
via correlation, verifies statistical outputs against implementation-independent mathematical
properties, and confirms branch isolation at the raw-column level.

- **File chain of custody:** raw TSV present, non-truncated (12,778,156 bytes), SHA-256
  recorded (`9e42b3a0…9a90c15`). **(Update 2026-07-04: this is no longer just recorded for
  future reference — `verify_practical_audit.py` now asserts the checksum matches on every
  run and fails the suite on mismatch, so undetected input drift is an active check, not a
  passive log.)**
- **Sample accounting (pure regex):** 84 columns (3 meta + 81 sample) exactly; no duplicate
  headers; all 81 sample columns matched by regex with **zero unmatched/silently-dropped
  columns**; `[index]` tags unique and complete 1..81.
- **Workbook (independent parse):** 81 rows; no duplicate well assignments (rules out a
  physical loading mixup); no duplicate sample labels; peptide_ug fully populated and positive.
- **Sample-identity screen:** no near-duplicate columns (max pairwise r=0.9966 < 0.999,
  i.e. no accidental copy-paste/duplicate injection); no sample poorly correlated with its own
  declared group (rules out gross mislabeling).
  **⚠ FLAG (not a failure):** the single highest-correlation pair among all 3,240 —
  `10W_KI8` vs `14W_WT10` (r=0.9966, 100th percentile) — is cross-genotype **and**
  cross-timepoint, which is exactly the pattern a swap/mislabel would produce. The pair
  still has real biological difference (7,774/9,769 proteins jointly detected, 682 vs 449
  uniquely detected), so this reads as coincidental plasma-proteome similarity rather than
  proof of an error, but it could not be *ruled out* from data alone.

  **Follow-up (2026-07-04) — plate map & run order checked:**
  - **Plate map** (`plasma_sample_information.xlsx`): `10week KI8` = well **F5**,
    `14week WT10` = well **F9** — same row, 4 columns apart, **not physically adjacent**
    (rules out adjacent-well carryover/splash contamination during liquid handling).
  - **Report column order** (`[N]` tag in the raw TSV): `10week KI8` = **[27]**,
    `14week WT10` = **[80]** — 53 positions apart, each embedded within a same-genotype
    block (KI columns 21–32, then WT columns 33–43 for 10W; same pattern for 14W). This
    ordering is genotype-grouped-then-numeric (KI before WT, ascending sample number) —
    i.e. it reads as a **name-sort applied by the Spectronaut report export**, not a
    timestamped acquisition log, so it should **not** be treated as true LC-MS injection
    order.
  - **No genuine run-order record exists in this project.** `input_data/` contains only the
    Spectronaut PG report (no per-injection timestamp/instrument-method log) and the
    single-sheet sample-information workbook (well + peptide µg only, no run-order column);
    every sample column shares one batch date (`20260529`) with no finer-grained sequence
    information. This limitation is disclosed rather than papered over.
  - **Conclusion on this item:** the two available, checkable proxies (plate adjacency,
    report column position) show **no proximity** that would support a physical
    cross-contamination or column-swap explanation. Combined with the real biological
    difference between the two samples, the anomalous correlation is most plausibly
    coincidental plasma-proteome similarity. It is **not fully resolvable** without an
    external record (e.g. the instrument's acquisition log / autosampler queue), which is
    outside this repository — recommend requesting that record from the acquisition core
    facility if absolute certainty is required before publication-grade interpretation.
- **Statistical-output correctness (implementation-independent):** BH-adjusted p-values are
  monotonic and `adj.P.Val ≥ P.Value` in all 20 contrasts (a property of the BH procedure
  itself, not of our code); `mean_KI_8W`/`mean_WT_8W` in `dep_results` match an **exact**
  recompute from `imputed_matrix.tsv` (not merely correlated — max abs diff ~1e-15).
- **Cross-branch isolation:** WT branch contains zero `_KI` columns and vice versa; the two
  branches' sample sets are disjoint at the raw-column-name level.

### On "raw p<0.05" as the significance criterion
This is a known, deliberate choice (set by the user for the DEP/volcano/trend calls), not a
defect — `adj.P.Val` is still computed and reported alongside every call. It is a real
**statistical-methodology** consideration (more false positives than an FDR-controlled
threshold), which is a different axis from **data integrity** (no corruption/scrambling).
This report only certifies the latter.

---

## Reproducing this verification
```bash
python3 scripts/run_pipeline.py && python3 scripts/run_branch.py   # regenerate outputs
bash tests/run_all.sh                                               # runs all 5 verify_*.py, fail-fast
```

## Conclusion
No data corruption, sample/protein scrambling, metadata mismatch, cross-branch contamination,
or silent data loss was found at any stage, using both pipeline-consistency checks (§§1–4) and
implementation-independent checks (§5). Every transformation reproduces exactly from the raw
inputs, sample↔group↔metadata mapping is exact against the source workbook, and the downstream
analytical outputs (DEP, pairwise, trend) are internally consistent and derive from the correct
data. **Data integrity is confirmed at the mechanical/pipeline level.**

One item remains open and is **not** resolvable from data alone: the `10W_KI8` / `14W_WT10`
sample pair is anomalously similar (§5) and should be checked against the physical run
records (plate map / LC-MS injection order) before being treated as fully independent in
downstream biological interpretation. This does not indicate a pipeline defect — it is a
sample-provenance question for the wet-lab/acquisition record, outside what code-level
verification can settle.
