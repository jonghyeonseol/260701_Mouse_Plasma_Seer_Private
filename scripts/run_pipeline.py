"""Pipeline orchestrator for the Mouse Plasma SEER proteomics analysis.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --dataset MousePlasma

Steps:
    01  raw_summary + QC (peptide recovery, plate batch)
    01b mouse-brain protein annotation (disk-only; does not alter step02+ input)
    02  filter (per-group valid fraction)
    03  log2 transform
    04  normalize (reference-ratio + median)
    05  impute (MinProb)
    06  DEP via limma (20 contrasts, ~0 + group + peptide_ug)
    07  dimensionality reduction (PCA / t-SNE / UMAP)  [integrated module]
    08  hierarchical clustering analysis (samples + features)  [integrated module]
    10  pairwise 2-group analysis (PCA/t-SNE/UMAP + top-DEP heatmap per pair)
    11  mouse-brain-protein-filtered re-analysis (dim-reduction + clustering)
    12  regionally-elevated-only re-analysis (763-gene DB subset -> real-data coverage)

Required last step after any regeneration:
    bash tests/run_all.sh    # data-integrity verification suite (must pass)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import utils
import step01_raw_summary as s1
import step01b_brain_annotation as s1b
import step02_filter as s2
import step03_log2_transform as s3
import step04_normalize as s4
import step05_impute as s5
import step06_dep_limma as s6
import step07_dim_reduction as s7
import step08_clustering as s8
import step10_pairwise as s10
import step11_brain_filtered as s11
import step12_regionally_elevated as s12


def _save_snapshot():
    import json
    os.makedirs(config.LOG_DIR, exist_ok=True)
    snap = {
        "timestamp": _dt.datetime.now().isoformat(),
        "VALID_FRACTION": config.VALID_FRACTION,
        "VALID_FRACTION_MODE": config.VALID_FRACTION_MODE,
        "NORMALIZE_METHOD": config.NORMALIZE_METHOD,
        "USE_PEPTIDE_COVARIATE": config.USE_PEPTIDE_COVARIATE,
        "LOG2FC_THRESHOLD": config.LOG2FC_THRESHOLD,
        "ADJ_PVALUE_THRESHOLD": config.ADJ_PVALUE_THRESHOLD,
        "IMPUTE_DOWNSHIFT": config.IMPUTE_DOWNSHIFT,
        "IMPUTE_WIDTH": config.IMPUTE_WIDTH,
        "IMPUTE_SEED": config.IMPUTE_SEED,
        "RANDOM_SEED": config.RANDOM_SEED,
        "GROUP_ORDER": config.PLASMA_GROUP_ORDER,
        "COMPARISONS": [{k: c[k] for k in ("name", "expr", "family")}
                        for c in config.COMPARISONS],
    }
    p = os.path.join(config.LOG_DIR, "pipeline_config_snapshot.json")
    with open(p, "w") as f:
        json.dump(snap, f, indent=2)
    return p


def run_dataset(dataset: str):
    logger = utils.setup_logger(dataset)
    logger.info(f"=== START pipeline {dataset} ===")
    state = s1.run(dataset, logger)
    state = s1b.run(dataset, logger, state)
    for step in (s2, s3, s4, s5, s6, s7, s8, s10, s11, s12):
        state = step.run(dataset, logger, state)
    logger.info(f"=== END pipeline {dataset} ===")
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(config.DATASETS.keys()) + ["all"],
                   default="MousePlasma")
    args = p.parse_args()
    _save_snapshot()
    targets = list(config.DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for d in targets:
        run_dataset(d)


if __name__ == "__main__":
    main()
