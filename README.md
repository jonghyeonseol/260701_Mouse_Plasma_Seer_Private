# 260701_Mouse_Plasma_Seer_Private

Mouse plasma proteomics analysis pipeline for a SEER Proteograph dataset — a
2-factor longitudinal design (genotype × timepoint) built on top of the
KBSI FAIMS reference pipeline (`KBSI_FAIMS_SC_260513`).

## Data

- **Input**: `input_data/20260630_142950_20260529_Report.tsv` — Spectronaut PG-level
  export, 9,769 protein groups × 81 quantity columns (80 biological samples + 1
  pooled reference), plus `input_data/plasma_sample_information.xlsx` (well
  location, peptide recovery µg).
- **Design**: genotype `{WT, KI}` × timepoint `{8, 10, 12, 14} weeks` → 8 groups
  (`WT_8W` … `KI_14W`), unbalanced replicate counts (8–12 per group).

## Pipeline structure

Three project scopes share the same step scripts (`scripts/step0*_*.py`,
`utils.py`, `config.py`):

| Project | Scope | Orchestrator | Output root |
|---|---|---|---|
| **Main** | all 8 groups | `python3 scripts/run_pipeline.py` | `output_data/MousePlasma/` |
| **WT branch** | WT only, 4 timepoints, re-filtered | `python3 scripts/run_branch.py --dataset MousePlasma_WT` | `output_data/MousePlasma_WT/` |
| **KI branch** | KI only, 4 timepoints, re-filtered | `python3 scripts/run_branch.py --dataset MousePlasma_KI` | `output_data/MousePlasma_KI/` |

`python3 scripts/run_branch.py` (no `--dataset`) runs both branches.

### Steps

| Step | Script | Main | Branch | Description |
|---|---|:---:|:---:|---|
| 01 | `step01_raw_summary.py` | ✅ | — | Raw QC: protein IDs/sample, CV distribution, missing-value heatmap, peptide-recovery vs depth, plate positional (batch) check |
| 02 | `step02_filter.py` | ✅ | ✅ | Per-group valid-fraction filter (≥50% detected in **every** group) |
| 03 | `step03_log2_transform.py` | ✅ | ✅ | log2 transform |
| 04 | `step04_normalize.py` | ✅ | ✅ | Reference-ratio (÷ pooled ref) + sample-wise median normalization |
| 05 | `step05_impute.py` | ✅ | ✅ | MinProb imputation (per-protein left-censored Gaussian downshift) |
| 06 | `step06_dep_limma.py` | ✅ | — | limma DEP: single means-model fit (`~0 + group + peptide_ug`), **20 contrasts** (see below), volcano plots, UpSet overlaps, union-DEP overview |
| 07 | `step07_dim_reduction.py` | ✅ | ✅ | PCA / t-SNE / UMAP, colored by design axes (branch: timepoint only) |
| 08 | `step08_clustering.py` | ✅ | ✅ | Sample correlation clustermap + dendrogram, ARI/purity vs design labels, top-variance & union-DEP feature heatmaps |
| 09 | `step09_trend_timeseries.py` | — | ✅ | Fuzzy c-means temporal-profile clustering + monotonic trend test (limma linear-week + Jonckheere–Terpstra + Spearman), R/ComplexHeatmap `trend_heatmap` |
| 10 | `step10_pairwise.py` | ✅ | — | Per-pair (2-group) PCA/t-SNE/UMAP + DEP/top-500-variance/total-feature heatmaps, silhouette separation summary |

### DEP contrast matrix (main, 20 contrasts)

| Family | Count | Contrasts |
|---|:---:|---|
| `genotype_at_time` | 4 | KI vs WT @ 8W, 10W, 12W, 14W |
| `time_course` | 12 | All pairwise timepoints within each genotype (6 per genotype) |
| `interaction` | 3 | (KI−WT)@t − (KI−WT)@8W, for t = 10/12/14W |
| `global` | 1 | KI(all) vs WT(all), time averaged out |

Significance: `raw p<0.05, |log2FC|>1.0` (config: `SIG_PVALUE_COL`/`PVALUE_THRESHOLD`/
`LOG2FC_THRESHOLD` in `scripts/config.py`); BH-adjusted p is always reported alongside.

## Key design choices

- **Normalization**: pooled-reference ratio (per-protein, log2 scale) followed by
  sample-median centering — anchors proteins to a common reference while removing
  per-sample loading differences.
- **Peptide covariate**: total µg peptide recovered is included as a limma design
  covariate (`peptide_ug`), since it correlates with proteome depth.
- **Branch re-filtering**: WT/KI branches re-apply the 50%-valid-fraction filter
  within their own 4-group subset (not inherited from the main 8-group filter),
  since the constraining group set changes.
- **z-score colormap**: green–black–red across all clustermaps/heatmaps
  (`config.ZSCORE_COLORS`).

## Setup

```bash
pip install -r requirements.txt
```

Requires R with the **limma** (Bioconductor), **ComplexHeatmap**, and **circlize**
packages installed, accessed via `rpy2` (used for DEP fitting and the trend
heatmap).

## Running

```bash
python3 scripts/run_pipeline.py           # main: preprocessing -> DEP -> dim-reduction -> clustering -> pairwise
python3 scripts/run_branch.py             # both branches: preprocessing -> dim-reduction -> clustering -> trend/time-series
```

## Verification

`tests/` holds a persisted data-integrity verification suite — each script
independently re-derives its target transform from the raw inputs and compares
to the pipeline's saved outputs (exits 0 on success):

```bash
bash tests/run_all.sh                     # runs all 5 scripts below, fail-fast
```

```bash
python3 tests/verify_preprocessing.py     # steps 01-05 (main), independent recompute
python3 tests/verify_integrity.py         # cross-step ID/order + metadata-join + sanity
python3 tests/verify_branch.py            # WT/KI branch subset + steps 02-05
python3 tests/verify_analysis.py          # DEP/pairwise/trend layer + completeness
python3 tests/verify_practical_audit.py   # implementation-independent audit (file
                                           # checksums, raw regex parsing, sample-
                                           # identity/duplication screening, BH math)
```

**`bash tests/run_all.sh` must be run (and must pass) before merging any change
to `scripts/`** — see `claudedocs/DATA_INTEGRITY_PM_PLAN.md` Phase B/C.

See `claudedocs/DATA_INTEGRITY_REPORT.md` for the full results and one open item
flagged for manual review (a high-correlation sample pair — not a proven defect,
see the report for details).

## Output layout

Each project root (`output_data/<Project>/`) contains one directory per step
(`01_raw_summary/` … `10_pairwise/`) plus a `figures/` directory mirroring every
plot in one place. See `claudedocs/DATA_INTEGRITY_REPORT.md` and the step
scripts for per-file descriptions.
