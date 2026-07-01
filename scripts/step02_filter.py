"""Step 02 — filter proteins by per-group valid fraction.

A protein is kept when ≥VALID_FRACTION of samples are non-missing in EVERY
biological group (VALID_FRACTION_MODE='all').  The pooled-ref column is not
part of any biological group and does not affect the filter, but its row values
are carried through for the reference-ratio normalization in step04.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

import config
import utils


def run(dataset: str, logger, state: dict) -> dict:
    cfg = config.DATASETS[dataset]
    out = utils.step_path(dataset, "02_filtered")
    utils.apply_style()

    df = state["data_df"].copy()
    meta = state["meta"]
    sample_cols = state["sample_cols"]           # biological only
    groups = cfg["group_order"]
    n0 = len(df)

    filtered_df, keep_mask, stats = utils.filter_valid_fraction(
        df, meta, sample_cols, groups, config.VALID_FRACTION,
        config.VALID_FRACTION_MODE)
    excluded = df[~keep_mask].copy()

    utils.save_tsv(filtered_df, out, "filtered_matrix")
    if not excluded.empty:
        utils.save_tsv(excluded[[c for c in cfg["meta_cols"] if c in excluded.columns]],
                       out, "filter_excluded_low_valid")

    stats.update({
        "dataset": dataset,
        "n_proteins_initial": n0,
        "valid_fraction_threshold": config.VALID_FRACTION,
        "valid_fraction_mode": config.VALID_FRACTION_MODE,
        "groups_considered": groups,
    })
    with open(os.path.join(out, "filter_log.txt"), "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    valid_label = (f"Valid≥{int(config.VALID_FRACTION*100)}% in "
                   f"{'ALL' if config.VALID_FRACTION_MODE == 'all' else 'ANY'} group")
    counts = [n0, stats["n_after"]]
    ax.bar(["Initial", valid_label], counts, color=["#BDBDBD", "#0072B2"],
           edgecolor="black", linewidth=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c + counts[0] * 0.02, f"{c:,}", ha="center", va="bottom", fontsize=10)
        if i > 0:
            ax.text(i, c * 0.5, f"Δ {c - counts[0]:+,}", ha="center", va="center",
                    fontsize=9, color="#444", fontweight="semibold")
    ax.set_ylim(0, counts[0] * 1.12)
    ax.set_ylabel("Proteins")
    ax.set_title(f"Filter summary — {dataset}")
    plt.xticks(rotation=15)
    utils.save_fig(fig, out, "filter_summary")
    plt.close(fig)

    logger.info(f"[step02] {dataset}: {n0} → {stats['n_after']} proteins "
                f"(removed {stats['n_removed']}).")
    state = dict(state)
    state["data_df"] = filtered_df
    return state


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import step01_raw_summary as s1
    log = utils.setup_logger("MousePlasma")
    run("MousePlasma", log, s1.run("MousePlasma", log))
