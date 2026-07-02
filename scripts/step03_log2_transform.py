"""Step 03 — log2 transformation of filtered intensities.

All sample columns (including the pooled reference) are log2-transformed so the
reference-ratio in step04 is a simple subtraction on the log2 scale.
Per-sample distributions are drawn for the biological samples only.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
import utils


def boxplot_by_sample(df, sample_cols, meta, group_order, out, name, title, *,
                      ylabel: str):
    """Per-sample log2 boxplot, ordered/coloured by biological group."""
    cols, ordered_meta = utils.ordered_bio_cols(meta, sample_cols, group_order)
    n = len(cols)
    width = min(28.0, max(8.0, 0.10 * n + 4.0))
    fig, ax = plt.subplots(figsize=(width, 5.5))
    colors = [config.GROUP_COLORS.get(g, "#888") for g in ordered_meta["group"]]
    data = [df[c].dropna().values for c in cols]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.65)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
        patch.set_edgecolor("black"); patch.set_linewidth(0.4)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(0.8)
    stride = max(1, int(np.ceil(n / 24)))
    ax.set_xticks(range(1, n + 1, stride))
    ax.set_xticklabels(ordered_meta["sample_id"].iloc[::stride].tolist(),
                       rotation=70, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    all_vals = np.concatenate([d for d in data if d.size > 0]) if data else np.array([])
    if all_vals.size:
        ax.axhline(float(np.nanmedian(all_vals)), color="#888", lw=0.6, ls="--")
    # Reserve headroom so the group sub-labels sit INSIDE the axes, below the
    # title (which is lifted with pad) — avoids title/subtitle overlap.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.10)
    ylabel_y = ymax + (ymax - ymin) * 0.015
    grp_seq = list(ordered_meta["group"])
    start, prev = 0, None
    for i, g in enumerate(grp_seq):
        if prev is not None and g != prev:
            ax.axvline(i + 0.5, color="#bbb", lw=0.7)
            ax.text((start + i) / 2 + 1, ylabel_y, config.GROUP_DISPLAY.get(prev, prev),
                    ha="center", va="bottom", fontsize=8, color="#333", weight="bold")
            start = i
        prev = g
    ax.text((start + n - 1) / 2 + 1, ylabel_y, config.GROUP_DISPLAY.get(prev, prev),
            ha="center", va="bottom", fontsize=8, color="#333", weight="bold")
    ax.set_title(title, pad=26)
    utils.save_fig(fig, out, name)
    plt.close(fig)


def density_by_group(df, sample_cols, meta, group_order, out, name, title, xlabel):
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    grp_map = meta.set_index("column_name")["group"]
    for c in sample_cols:
        g = grp_map.get(c)
        vals = df[c].dropna().values
        if len(vals) < 20:
            continue
        ax.hist(vals, bins=60, histtype="step",
                color=config.GROUP_COLORS.get(g, "#888"), alpha=0.3, linewidth=0.6)
    handles = [mpatches.Patch(color=config.GROUP_COLORS[g], label=config.GROUP_DISPLAY[g])
               for g in group_order]
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=2, fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel("Frequency (per sample)")
    ax.set_title(title)
    utils.save_fig(fig, out, name)
    plt.close(fig)


def run(dataset: str, logger, state: dict) -> dict:
    cfg = config.DATASETS[dataset]
    out = utils.step_path(dataset, "03_log2_transformed")
    utils.apply_style()

    df = state["data_df"].copy()
    sample_cols = state["sample_cols"]
    ref_col = state.get("ref_col")
    meta = state["meta"]

    log_cols = list(sample_cols) + ([ref_col] if ref_col else [])
    df[log_cols] = np.log2(df[log_cols].astype(float))

    utils.save_tsv(df, out, "log2_matrix")
    boxplot_by_sample(df, sample_cols, meta, cfg["group_order"], out,
                      "boxplot_before_normalization",
                      f"Sample log2 distributions — before normalization ({dataset})",
                      ylabel="log2(intensity)")
    density_by_group(df, sample_cols, meta, cfg["group_order"], out,
                     "distribution_before",
                     f"Log2 intensity distribution — before normalization ({dataset})",
                     "log2(intensity)")

    logger.info(f"[step03] {dataset}: log2 transform applied "
                f"(n_proteins={len(df)}, n_bio_samples={len(sample_cols)}).")
    state = dict(state)
    state["data_df"] = df
    state["log2_pre_norm_df"] = df.copy()
    return state


if __name__ == "__main__":
    pass
