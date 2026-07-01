"""Volcano plot for DEP results (mouse plasma SEER).

Adapted from KBSI_FAIMS_SC_260513/viz/viz_volcano.py.  Uses this dataset's
protein-ID column (config.ID_COL) and a precomputed ``label`` column (protein
display name from PG.ProteinNames) for point annotation.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import config
import utils

try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None


_COLORS = {"Up": "#FF3F3F", "Down": "#3232FF", "NS": "#95A5A6"}


def _direction_class(row, pcol, pthr, lfc):
    if row[pcol] >= pthr or abs(row["logFC"]) <= lfc:
        return "NS"
    return "Up" if row["logFC"] > 0 else "Down"


def plot_volcano(df: pd.DataFrame, *, comparison_name: str, output_dir: str,
                 case_grp: str, ctrl_grp: str,
                 pcol: str = None, pthr: float = None,
                 lfc: float = config.LOG2FC_THRESHOLD,
                 top_n: int = 15, figures_mirror: Optional[str] = None) -> str:
    utils.apply_style()
    # Significance column + threshold: raw P.Value by default (config-driven).
    pcol = pcol or config.SIG_PVALUE_COL
    pthr = config.PVALUE_THRESHOLD if pthr is None else pthr
    work = df.copy()
    work["neg_log10"] = -np.log10(work[pcol].clip(lower=1e-300))
    work["dclass"] = work.apply(lambda r: _direction_class(r, pcol, pthr, lfc), axis=1)
    n_up = int((work["dclass"] == "Up").sum())
    n_dn = int((work["dclass"] == "Down").sum())

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in ("NS", "Down", "Up"):
        sub = work[work["dclass"] == cls]
        ax.scatter(sub["logFC"], sub["neg_log10"], c=_COLORS[cls], s=42,
                   alpha=0.7, linewidths=0)
    ax.axhline(-np.log10(pthr), color="gray", ls="--", lw=1, alpha=0.8)
    ax.axvline(lfc, color="gray", ls="--", lw=1, alpha=0.8)
    ax.axvline(-lfc, color="gray", ls="--", lw=1, alpha=0.8)

    if top_n > 0 and (n_up + n_dn) > 0:
        sig = work[work["dclass"] != "NS"].copy()
        sig = sig.reindex(sig["logFC"].abs().sort_values(ascending=False).index).head(top_n)
        texts = []
        for _, row in sig.iterrows():
            label = str(row.get("label") or row[config.ID_COL]).split(";")[0]
            if len(label) > 18:
                label = label[:15] + "..."
            texts.append(ax.text(row["logFC"], row["neg_log10"], label,
                                  fontsize=10, alpha=0.9, ha="left", va="bottom"))
        if adjust_text is not None and texts:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5),
                        only_move={"points": "y", "text": "xy"})

    ax.set_xlabel(f"Log2 Fold Change ({case_grp} / {ctrl_grp})", fontsize=16)
    ax.set_ylabel(f"-Log10({config.sig_pvalue_axis_label()})", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_title(f"Volcano: {comparison_name}\n"
                 f"Features: {len(work)} | DEPs: {n_up + n_dn} "
                 f"(↑{n_up}, ↓{n_dn})", fontsize=12)
    ax.legend(handles=[Patch(facecolor=_COLORS["Up"], label=f"Up ({n_up})"),
                       Patch(facecolor=_COLORS["Down"], label=f"Down ({n_dn})"),
                       Patch(facecolor=_COLORS["NS"], label="Not significant")],
              loc="upper right", frameon=True, framealpha=1.0, edgecolor="#CFCFCF",
              fontsize=9)

    safe = re.sub(r"[^A-Za-z0-9]+", "_", comparison_name).strip("_")
    png = utils.save_fig(fig, output_dir, f"volcano_{safe}")
    plt.close(fig)
    if figures_mirror:
        utils.ensure_dir(figures_mirror)
        for p in (png, png.replace(".png", ".svg")):
            if os.path.exists(p):
                shutil.copy(p, os.path.join(figures_mirror, os.path.basename(p)))
    return png
