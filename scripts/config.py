"""Central configuration for the Mouse Plasma SEER proteomics pipeline.

Adapted from the KBSI FAIMS NK-cell single-cell pipeline
(KBSI_FAIMS_SC_260513) for a **bulk mouse plasma, SEER Proteograph** dataset
with a 2-factor longitudinal design:

  genotype  ∈ {WT, KI}
  timepoint ∈ {8, 10, 12, 14} weeks
  → 8 biological groups (genotype × timepoint), unbalanced sample sizes,
    plus one "Total plasma pooled" reference sample used for
    reference-ratio normalization and QC only (never a statistical group).

Key differences from the reference project (documented for maintainers):
  - Metadata columns are PG.ProteinGroups / PG.ProteinDescriptions /
    PG.ProteinNames.  There is **no** PG.Genes column, so a display label is
    derived from PG.ProteinNames (UniProt entry name, e.g. "ALBU_MOUSE").
  - Single-cell-only logic (per-cell QC, focus sub-clustering, depth
    correction) is dropped; sample-level QC is used instead.
  - Normalization is reference-ratio (÷ pooled ref, on log2 scale) followed by
    sample-wise median centering.
  - DEP uses ONE limma means-model fit (~0 + group + peptide_ug) from which all
    14 contrasts (4 analysis goals) are extracted via makeContrasts, so eBayes
    shares variance across the whole design.
"""

import os
import re

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "input_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_data")
SHARED_DIR = os.path.join(OUTPUT_DIR, "_shared")
LOG_DIR = os.path.join(SHARED_DIR, "logs")

PLASMA_INPUT = os.path.join(INPUT_DIR, "20260630_142950_20260529_Report.tsv")
# Sample information workbook (well location + peptide recovery).
SAMPLE_INFO_XLSX = os.path.join(INPUT_DIR, "plasma_sample_information.xlsx")
PLASMA_DIR = os.path.join(OUTPUT_DIR, "MousePlasma")

DATABASE_DIR = os.path.join(BASE_DIR, "database")
# HPA export: human gene records (Gene/Ensembl/Uniprot = human) carrying mouse-brain
# RNA regional-specificity data (see scripts/step01b_brain_annotation.py). Every row
# has a non-null "RNA mouse brain regional specificity" — the file itself is already
# a curated "detected in mouse brain" gene list, so membership = mouse-brain protein.
MOUSE_BRAIN_DB = os.path.join(DATABASE_DIR, "260525_v25.1_mouse_brain_category_rna_Any_Region.tsv")
BRAIN_DB_GENE_COL = "Gene"
BRAIN_DB_REGIONAL_COL = "RNA mouse brain regional specificity"
# "Elevated" = HPA's standard union of enriched/enhanced categories, excluding
# "Low region specificity" (confirmed with user 2026-07-13).
BRAIN_ELEVATED_CATEGORIES = ["Region enriched", "Region enhanced", "Group enriched"]

# Cache of mouse UniProt accession -> UniProt gene-name synonyms (space-separated),
# fetched via the UniProt REST API (see step01b). Avoids re-querying on every run.
UNIPROT_GENE_NAME_CACHE = os.path.join(SHARED_DIR, "uniprot_gene_names_cache.tsv")

STEP_DIRS = {
    "01_raw_summary":      "01_raw_summary",
    "02_filtered":         "02_filtered",
    "03_log2_transformed": "03_log2_transformed",
    "04_normalized":       "04_normalized",
    "05_imputed":          "05_imputed",
    "06_dep_limma":        "06_dep_limma",
    "07_dim_reduction":    "07_dim_reduction",
    "08_clustering":       "08_clustering",
    "09_trend_timeseries": "09_trend_timeseries",
    "10_pairwise":         "10_pairwise",
    "11_brain_filtered":   "11_brain_filtered",
    "12_regionally_elevated": "12_regionally_elevated",
    "figures":             "figures",
}

# ── Metadata / ID columns (Spectronaut PG export for this dataset) ────────────
PLASMA_META_COLS = ["PG.ProteinGroups", "PG.ProteinDescriptions", "PG.ProteinNames"]
ID_COL = "PG.ProteinGroups"          # primary protein-group identifier
NAME_COL = "PG.ProteinNames"         # UniProt entry names → display label source
DESC_COL = "PG.ProteinDescriptions"

# ── Sample column parsing ────────────────────────────────────────────────────
# Biological sample columns look like:
#   "[1] 20260529_SEER_HanKH_mouse_plasma_8W_KI1.raw.PG.Quantity"
# capture: index (1), date (20260529), weeks (8), genotype (KI), number (1)
PLASMA_COL_PATTERN = re.compile(
    r"\[(\d+)\]\s+(\d{8})_SEER_HanKH_mouse_plasma_(\d+)W_(WT|KI)(\d+)\.raw\.PG\.Quantity"
)
# Pooled reference column:
#   "[81] 20260529_SEER_HanKH_mouse_plasma_Total_pooled_ref.raw.PG.Quantity"
PLASMA_REF_PATTERN = re.compile(
    r"\[(\d+)\]\s+(\d{8})_SEER_HanKH_mouse_plasma_Total_pooled_ref\.raw\.PG\.Quantity"
)
REF_GROUP = "pooled_ref"             # group label carried by the pooled sample

# ── Experimental design ──────────────────────────────────────────────────────
GENOTYPES = ["WT", "KI"]
TIMEPOINTS = [8, 10, 12, 14]         # weeks; 8W is the time-course baseline
TIME_BASELINE = 8

# 8 biological groups in canonical order (WT block first, then KI, time ascending)
PLASMA_GROUP_ORDER = [f"{g}_{t}W" for g in GENOTYPES for t in TIMEPOINTS]

GROUP_DISPLAY = {f"{g}_{t}W": f"{g} {t}W" for g in GENOTYPES for t in TIMEPOINTS}
GROUP_DISPLAY[REF_GROUP] = "Pooled ref"

# Per-group colors: WT = blue gradient (light→dark with time),
#                   KI = red/orange gradient (light→dark with time).
_WT_SHADES = ["#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]
_KI_SHADES = ["#FCAE91", "#FB6A4A", "#DE2D26", "#A50F15"]
PLASMA_GROUP_COLORS = {}
for _t, _c in zip(TIMEPOINTS, _WT_SHADES):
    PLASMA_GROUP_COLORS[f"WT_{_t}W"] = _c
for _t, _c in zip(TIMEPOINTS, _KI_SHADES):
    PLASMA_GROUP_COLORS[f"KI_{_t}W"] = _c
PLASMA_GROUP_COLORS[REF_GROUP] = "#7F8C8D"
GROUP_COLORS = dict(PLASMA_GROUP_COLORS)

# Single-factor color maps for dim-reduction / QC / heatmap annotation.
GENOTYPE_COLORS = {"WT": "#AFDCEC", "KI": "#FF4040"}
TIMEPOINT_COLORS = {8: "#440154", 10: "#31688E", 12: "#35B779", 14: "#FDE725"}

# Diverging z-score colormap for expression heatmaps: softened green → black → red
# (classic microarray palette; black midpoint at z≈0).
ZSCORE_COLORS = ["#00C800", "#000000", "#E00000"]

# Direction colors for DEP / volcano / heatmap row annotations.
DIRECTION_COLORS = {"Up": "#C0392B", "Down": "#1F77B4", "NS": "#BDBDBD"}

# ── DEP contrast specification (4 analysis goals → 20 contrasts) ──────────────
# Each contrast is expressed in terms of the 8 group means so a single limma
# means-model fit (~0 + group [+ peptide]) yields every comparison via
# makeContrasts.  case_grp / ctrl_grp drive volcano axis labels and the sign
# convention (positive logFC = higher in case).
import itertools as _it


def _mk(name, expr, case, ctrl, family):
    return {"name": name, "expr": expr, "case_grp": case,
            "ctrl_grp": ctrl, "family": family}


COMPARISONS = []
# Goal 1 — timepoint-wise KI vs WT (4)
for _t in TIMEPOINTS:
    COMPARISONS.append(_mk(
        f"KI_vs_WT_{_t}W", f"KI_{_t}W - WT_{_t}W",
        f"KI_{_t}W", f"WT_{_t}W", "genotype_at_time"))
# Goal 2 — within-genotype time course, ALL pairwise timepoints (12).
# Convention: later timepoint = case, earlier = ctrl (positive logFC = increase
# over time).  6 ordered pairs per genotype (8-10, 8-12, 8-14, 10-12, 10-14, 12-14).
for _g in GENOTYPES:
    for _earlier, _later in _it.combinations(TIMEPOINTS, 2):
        COMPARISONS.append(_mk(
            f"{_g}_{_later}W_vs_{_earlier}W", f"{_g}_{_later}W - {_g}_{_earlier}W",
            f"{_g}_{_later}W", f"{_g}_{_earlier}W", "time_course"))
# Goal 3 — genotype × time interaction, each time vs 8W baseline (3)
for _t in TIMEPOINTS:
    if _t == TIME_BASELINE:
        continue
    COMPARISONS.append(_mk(
        f"interaction_{_t}W_vs_{TIME_BASELINE}W",
        f"(KI_{_t}W - WT_{_t}W) - (KI_{TIME_BASELINE}W - WT_{TIME_BASELINE}W)",
        f"KI-WT@{_t}W", f"KI-WT@{TIME_BASELINE}W", "interaction"))
# Goal 4 — global genotype main effect (time averaged out) (1)
_ki_mean = "(" + " + ".join(f"KI_{t}W" for t in TIMEPOINTS) + f")/{len(TIMEPOINTS)}"
_wt_mean = "(" + " + ".join(f"WT_{t}W" for t in TIMEPOINTS) + f")/{len(TIMEPOINTS)}"
COMPARISONS.append(_mk(
    "KI_vs_WT_global", f"{_ki_mean} - {_wt_mean}",
    "KI(all)", "WT(all)", "global"))

# Contrasts used for the pairwise-style cross-contrast summary plots (goal 1).
GENOTYPE_AT_TIME = [c for c in COMPARISONS if c["family"] == "genotype_at_time"]

# ── Statistical thresholds ───────────────────────────────────────────────────
VALID_FRACTION = 0.50
VALID_FRACTION_MODE = "all"          # ≥50% valid in EVERY biological group
LOG2FC_THRESHOLD = 1.0
ADJ_PVALUE_THRESHOLD = 0.05          # BH-adjusted p (reported column, kept for reference)

# Significance is evaluated on SIG_PVALUE_COL at PVALUE_THRESHOLD, and the
# volcano y-axis uses the same column.  Per user directive this is the RAW
# p-value ("P.Value").  NOTE: raw p<0.05 applies no multiple-testing correction,
# so it is more permissive (higher false-positive rate) than BH-adjusted p; the
# BH-adjusted value is still reported in every results table as 'adj.P.Val'.
SIG_PVALUE_COL = "P.Value"           # "P.Value" (raw) or "adj.P.Val" (BH)
PVALUE_THRESHOLD = 0.05


def sig_criterion_label() -> str:
    """Human-readable DEP criterion for figure titles / summaries."""
    kind = "raw p" if SIG_PVALUE_COL == "P.Value" else "adj.p"
    return f"{kind}<{PVALUE_THRESHOLD}, |log2FC|>{LOG2FC_THRESHOLD}"


def sig_pvalue_axis_label() -> str:
    return "Raw P-value" if SIG_PVALUE_COL == "P.Value" else "Adjusted P-value"

# limma model: include peptide-recovery (µg) as a numeric covariate.
USE_PEPTIDE_COVARIATE = True
PEPTIDE_COL = "peptide_ug"

# ── Normalization ────────────────────────────────────────────────────────────
# "ref_ratio_median": subtract pooled-ref log2 per protein (reference ratio),
# then subtract per-sample median (loading correction).
NORMALIZE_METHOD = "ref_ratio_median"

# ── MinProb imputation ───────────────────────────────────────────────────────
IMPUTE_DOWNSHIFT = 1.8
IMPUTE_WIDTH = 0.3
IMPUTE_SEED = 42

# ── Plot settings ────────────────────────────────────────────────────────────
DPI = 300
FONT_FAMILY = "Arial"
RANDOM_SEED = 42

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 11
LEGEND_SIZE = 11

FIGSIZE_SINGLE = (7.5, 6.0)
FIGSIZE_WIDE = (12.0, 5.0)
FIGSIZE_SQUARE = (7.0, 7.0)
FIGSIZE_VOLCANO = (8.5, 7.0)

HEATMAP_ZSCORE_RANGE = (-3, 3)

# ── Trend / time-series analysis (branch step09) ─────────────────────────────
TREND_N_CLUSTERS = 6          # fuzzy c-means clusters for temporal profiles
TREND_FUZZIFIER = 1.5         # c-means fuzzifier m (Mfuzz-style soft membership)
TREND_SEED = 42
TREND_CORE_MEMBERSHIP = 0.5   # membership ≥ this = "core" cluster member
TREND_HEATMAP_TOP_N = 30      # top-N per direction shown (labeled) in trend_heatmap

# ── Dataset registry: main (all 8 groups) + per-genotype branches ────────────
# Branch datasets subset to one genotype's 4 timepoint groups (+ pooled ref for
# normalization); the 50% valid-fraction filter is therefore re-applied within
# the subset.  dimred_axes / annotation_axes drive which design axes the
# dim-reduction colours and the clustermap annotation strips use.
_branch_groups = {g: [f"{g}_{t}W" for t in TIMEPOINTS] for g in GENOTYPES}
_branch_colors = {g: {grp: PLASMA_GROUP_COLORS[grp] for grp in _branch_groups[g]}
                  for g in GENOTYPES}

DATASETS = {
    "MousePlasma": {
        "input":          PLASMA_INPUT,
        "sample_info":    SAMPLE_INFO_XLSX,
        "output_root":    PLASMA_DIR,
        "meta_cols":      PLASMA_META_COLS,
        "group_order":    PLASMA_GROUP_ORDER,
        "normalize_method": NORMALIZE_METHOD,
        "comparisons":    COMPARISONS,
        "group_colors":   PLASMA_GROUP_COLORS,
        "genotype":       None,                # full cohort
        "timepoints":     TIMEPOINTS,
        "dimred_axes":    ["genotype", "timepoint", "group"],
        "annotation_axes": ["Genotype", "Timepoint", "Group"],
    },
}
for _g in GENOTYPES:
    DATASETS[f"MousePlasma_{_g}"] = {
        "input":          PLASMA_INPUT,
        "sample_info":    SAMPLE_INFO_XLSX,
        "output_root":    os.path.join(OUTPUT_DIR, f"MousePlasma_{_g}"),
        "meta_cols":      PLASMA_META_COLS,
        "group_order":    _branch_groups[_g],
        "normalize_method": NORMALIZE_METHOD,
        "comparisons":    [],                  # no DEP in branch projects
        "group_colors":   _branch_colors[_g],
        "genotype":       _g,                  # subset selector
        "timepoints":     TIMEPOINTS,
        "dimred_axes":    ["timepoint"],       # genotype is constant in a branch
        "annotation_axes": ["Timepoint"],
    }
