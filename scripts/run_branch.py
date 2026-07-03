"""Branch-project orchestrator — per-genotype subsets (WT-only / KI-only).

Each branch re-runs preprocessing on ONE genotype's samples (+ pooled ref), so
the 50%% valid-fraction filter is re-applied within the 4 timepoint groups, then
runs dim-reduction, hierarchical clustering, and trend / time-series analysis.
DEP (step06) and the heavy step01 QC are intentionally skipped.

Usage:
    python scripts/run_branch.py
    python scripts/run_branch.py --dataset MousePlasma_WT

Required last step after any regeneration:
    bash tests/run_all.sh    # data-integrity verification suite (must pass)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import utils
import step02_filter as s2
import step03_log2_transform as s3
import step04_normalize as s4
import step05_impute as s5
import step07_dim_reduction as s7
import step08_clustering as s8
import step09_trend_timeseries as s9


BRANCH_DATASETS = [k for k in config.DATASETS if k != "MousePlasma"]


def _initial_state(dataset: str, logger) -> dict:
    """Build a step01-equivalent state for a genotype subset (skips heavy QC)."""
    cfg = config.DATASETS[dataset]
    genotype = cfg["genotype"]
    df, meta = utils.load_raw(dataset)
    all_cols = meta["column_name"].tolist()
    df = utils.to_numeric_intensity(df, all_cols)
    sub_df, sub_meta, sample_cols, ref_col = utils.subset_by_genotype(df, meta, genotype)
    logger.info(f"[branch:{dataset}] subset genotype={genotype}: "
                f"{sub_df.shape[0]} proteins × {len(sample_cols)} samples "
                f"(+{'1 pooled ref' if ref_col else '0 ref'}).")
    return {"data_df": sub_df, "meta": sub_meta, "sample_cols": sample_cols,
            "ref_col": ref_col, "all_sample_cols": sample_cols + ([ref_col] if ref_col else [])}


def _save_snapshot(dataset: str):
    cfg = config.DATASETS[dataset]
    os.makedirs(config.LOG_DIR, exist_ok=True)
    snap = {
        "timestamp": _dt.datetime.now().isoformat(),
        "dataset": dataset,
        "genotype": cfg["genotype"],
        "group_order": cfg["group_order"],
        "NORMALIZE_METHOD": config.NORMALIZE_METHOD,
        "VALID_FRACTION": config.VALID_FRACTION,
        "TREND_N_CLUSTERS": config.TREND_N_CLUSTERS,
        "TREND_FUZZIFIER": config.TREND_FUZZIFIER,
    }
    with open(os.path.join(config.LOG_DIR, f"branch_snapshot_{dataset}.json"), "w") as f:
        json.dump(snap, f, indent=2)


def run_dataset(dataset: str):
    logger = utils.setup_logger(dataset)
    logger.info(f"=== START branch pipeline {dataset} ===")
    _save_snapshot(dataset)
    state = _initial_state(dataset, logger)
    for step in (s2, s3, s4, s5, s7, s8, s9):
        state = step.run(dataset, logger, state)
    logger.info(f"=== END branch pipeline {dataset} ===")
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=BRANCH_DATASETS + ["all"], default="all")
    args = p.parse_args()
    targets = BRANCH_DATASETS if args.dataset == "all" else [args.dataset]
    for d in targets:
        run_dataset(d)


if __name__ == "__main__":
    main()
