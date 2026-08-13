"""trends_common.py

Shared definitions for run_ev_trend_analysis.py.

Mirrors branch/scripts/trends_common.R from the reference project
(260126_Data_Processing_Private): same ordered-stage color ramp, direction /
FDR-star colors, and the diverging heatmap color function, so the EV protein
trend figures read as the same visual system as the reference project's
06_Trend_Analysis outputs. The single ordered "disease stage" axis there
(Normal -> LGD/IGD -> HGD/CIS -> Inv.IPMN) is replaced here by the ordered
"timepoint" axis (8W -> 10W -> 12W -> 14W) within one genotype branch
(WT or KI), since this dataset's design is genotype x time rather than a
single disease-stage progression.
"""

from __future__ import annotations

import os

BRANCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BRANCH_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
DATABASE_DIR = os.path.join(PROJECT_ROOT, "database")

TSV_ROOT = os.path.join(BRANCH_DIR, "output", "tsv")
FIGURES_ROOT_BASE = os.path.join(BRANCH_DIR, "output", "figures")
RESULTS_DIR = os.path.join(BRANCH_DIR, "output", "results")

# Numbered audit-trail stages for THIS branch pipeline (candidate matching ->
# filter -> log2 -> normalize -> impute -> trend test), so every folder number
# reflects this pipeline's own actual step order — not copied from the
# reference project's unrelated step numbering. These tsv/ stage folders are
# numbered 01-05 (candidate matching through imputation).
STAGE_01_CANDIDATE = "01_EV_Candidate_Matching"
STAGE_02_FILTERED = "02_Filtered"
STAGE_03_LOG2 = "03_Log2_Transformed"
STAGE_04_NORMALIZED = "04_Normalized"
STAGE_05_IMPUTED = "05_Imputed"

# figures/ only ever gets TWO stage folders (steps 02-05 write tsv audit
# trails only, no plots), so its own local sequence is 01_EV_Candidate_
# Matching, then 02_Trend_Analysis — the trend-analysis figures are the
# SECOND folder actually created under figures/, not the sixth.
STAGE_02_TREND = "02_Trend_Analysis"

TSV_DIR = TSV_ROOT  # backward-compat alias (unused by new stage-scoped writers)
FIGURES_ROOT = os.path.join(FIGURES_ROOT_BASE, STAGE_02_TREND)

EV_DB_PATH = os.path.join(DATABASE_DIR, "260812_mousebrain_elevated_exosome_database.tsv")


def tsv_stage_dir(stage: str) -> str:
    d = os.path.join(TSV_ROOT, stage)
    os.makedirs(d, exist_ok=True)
    return d


def figures_stage_dir(stage: str, genotype: str | None = None) -> str:
    d = os.path.join(FIGURES_ROOT_BASE, stage) if genotype is None else \
        os.path.join(FIGURES_ROOT_BASE, stage, genotype)
    os.makedirs(d, exist_ok=True)
    return d

# ── Ordered stage (timepoint) definitions & aesthetics ─────────────────────
# Colors copied 1:1 from the reference project's trends_common.R GROUP_COLORS
# (Normal=green, LGD/IGD=blue, HGD/CIS=orange, Inv.IPMN=red), remapped onto
# our 4 ordered timepoints so the stage-progression color ramp matches.
STAGE_ORDER = [8, 10, 12, 14]
STAGE_LABELS = {8: "8W", 10: "10W", 12: "12W", 14: "14W"}
STAGE_COLORS = {
    8: "#2ecc71",
    10: "#3498db",
    12: "#f39c12",
    14: "#e74c3c",
}
PHASE_MAP = {t: i + 1 for i, t in enumerate(STAGE_ORDER)}

FDR_THRESHOLD = 0.05

DIRECTION_COLORS = {"Increasing": "#e74c3c", "Decreasing": "#3498db", "Flat": "grey70"}
FDR_STAR_COLORS = {"***": "#8B0000", "**": "#CD5C5C", "*": "#F4A460", "ns": "grey85"}

# Blue-White-Red 5-stop diverging heatmap ramp — identical stops to the
# reference project's heatmap_col_fun() in trends_common.R.
HEATMAP_STOPS = [-2, -1, 0, 1, 2]
HEATMAP_COLORS = ["#3B4CC0", "#9BB8E3", "#F7F7F7", "#E88482", "#B40426"]

GENOTYPES = ["WT", "KI"]


def ensure_dirs():
    for d in (TSV_ROOT, RESULTS_DIR, FIGURES_ROOT):
        os.makedirs(d, exist_ok=True)
    for g in GENOTYPES:
        os.makedirs(os.path.join(FIGURES_ROOT, g), exist_ok=True)


def sig_star(fdr: float) -> str:
    if fdr < 0.001:
        return "***"
    if fdr < 0.01:
        return "**"
    if fdr < 0.05:
        return "*"
    return "ns"
