#!/usr/bin/env python3
"""EXPERIMENT — the merged three-panel story figure (fig3 + fig5 in one float).

Not part of the paper build. Standalone so make_figures.py stays untouched and
this whole experiment reverts with `rm`.

The point of merging is a shared y-order. Today fig3's two panels run in
OPPOSITE directions (panel a best-at-bottom, panel b best-at-top) and their
categories do not match 1:1 (a splits slope/fALFF, b merges them; b has
"All explicit" with no counterpart in a). So a reader cannot track a family
across the panels, which is the only thing the figure exists to let them do.

Here all three panels carry the same four families in the same order, sorted by
held-out r descending. That makes (b) and (c) clean descending staircases which
agree with each other, and (a) the odd one out -- Global FC is the bottom row
with the longest bar in (a) and nothing in (c). The dissociation is one
horizontal read.

Panel (c) additionally carries the two reference points (the 512-d model, and
the 3 features the attribution nominated) below a rule. They have no probe or
distillation counterpart, so they cannot be rows in (a) or (b); the space that
frees in those panels holds their legends.

Usage:
    python paper/hcp_ya/figures/make_fig_story.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "hcp_ya" / "brainlm"
INT = RES / "interpretability"
OUT = Path(__file__).resolve().parent

# Matched to make_figures.py so the merged figure sits beside the others.
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

COG = "#1f5f8b"   # connectivity families
HL = "#c1440e"    # timescale families
AFF = "#b0b0b0"


def load(split: str = "all") -> pd.DataFrame:
    """One row per family, with its encoding, use and held-out number."""
    pr = pd.read_csv(INT / "probe_signal_features.csv")
    pr = pr[pr.source == "frozen_mean"]
    fam_r2 = pr.groupby("family").r2.mean()
    # Panel (b)'s "spectral" set is slope + fALFF (14 features), so panel (a)'s
    # counterpart has to be the mean of those two probe families, not either
    # one alone. Equal n (7 each), so the unweighted mean is the right pooling.
    enc = {
        "spectral": float(fam_r2[["slope", "falff"]].mean()),
        "ac1": float(fam_r2["ac1"]),
        "netfc": float(fam_r2["netfc"]),
        "fc_global": float(fam_r2["fc"]),
    }

    # split="train" reads the distillation rerun on the 656 training subjects
    # (distill_hcp.py --split train); "all" keeps the full-cohort table the
    # original manuscript's figure was drawn from.
    sfx = "_train" if split == "train" else ""
    di = pd.read_csv(RES / "metrics" / f"distillation_PMAT24_A_CR{sfx}.csv").set_index("feature_set")
    cf = pd.read_csv(RES / "metrics" / f"distill_confirm_PMAT24_A_CR.csv").set_index("feature_set")

    rows = []
    for key, label, colour in [
        ("spectral", "Spectral (14)", HL),
        ("ac1", "Lag-1 autocorr. (7)", HL),
        ("netfc", "Network FC (28)", COG),
        ("fc_global", "Global FC (2)", COG),
    ]:
        rows.append({
            "key": key, "label": label, "colour": colour,
            "encoding_r2": enc[key],
            "pct_of_model": float(di.loc[key, "frac_of_model"]) * 100,
            "holdout_r": float(cf.loc[key, "holdout_r"]),
            "lo": float(cf.loc[key, "holdout_r_lo"]),
            "hi": float(cf.loc[key, "holdout_r_hi"]),
            "p": float(cf.loc[key, "holdout_r_p"]),
        })
    d = pd.DataFrame(rows).sort_values("holdout_r", ascending=False).reset_index(drop=True)
    return d, cf


def make(path: Path = OUT / "fig_story.pdf", split: str = "all") -> None:
    d, cf = load(split)
    n = len(d)
    y = np.arange(n)[::-1]            # row 0 at the top

    # Reference rows, panel (c) only, below a rule. Kept as tight as the labels
    # allow: this block is empty in (a) and (b), so every unit of height it takes
    # is height those two panels spend on nothing.
    y_rule = -0.55
    y_model, y_nom = -0.98, -1.58
    ylim = (-1.95, n - 0.45)

    fig, (a, b, c) = plt.subplots(
        1, 3, figsize=(5.9, 2.45),
        gridspec_kw={"width_ratios": [1.0, 0.86, 1.06], "wspace": 0.16})

    # --- (a) what the representation encodes ------------------------------
    a.barh(y, d.encoding_r2, color=d.colour, height=0.62)
    a.set_yticks(y, d.label)
    a.set_xlim(0, 1.0)
    a.set_xlabel("out-of-fold $R^2$")
    a.set_title("(a) What it encodes", loc="left")
    for yy, v in zip(y, d.encoding_r2):
        a.text(v - 0.03, yy, f"{v:.2f}", va="center", ha="right",
               fontsize=6.3, color="white")

    # --- (b) what the prediction uses -------------------------------------
    b.barh(y, d.pct_of_model, color=d.colour, height=0.62)
    b.set_yticks(y, [])
    b.set_xlim(0, 108)
    b.set_xlabel("% of the model's $R^2$")
    b.set_title("(b) What it uses", loc="left")
    for yy, v in zip(y, d.pct_of_model):
        b.text(v + 2.5, yy, f"{v:.0f}%", va="center", fontsize=6.3)

    # --- (c) what generalizes ---------------------------------------------
    for yy, r in zip(y, d.itertuples()):
        c.plot([r.lo, r.hi], [yy, yy], color=r.colour, lw=1.4,
               solid_capstyle="round", zorder=3)
        c.scatter([r.holdout_r], [yy], color=r.colour, s=22, zorder=4)
    # Global FC is the whole point of the panel, so its failure is labelled
    # rather than left to the reader to read off the interval.
    worst = d.iloc[-1]
    c.text(worst.hi + 0.018, y[-1], f"$p$ = {worst.p:.2f}", va="center",
           fontsize=6.3, color="0.35")

    # Labels sit to the LEFT of their intervals: to the right they would run off
    # the panel, and the space left of r = 0.09 is empty in both reference rows.
    ref = [(y_model, "BrainLM (512)", "BrainLM frozen embedding (the model)",
            "#333333", "D"),
           (y_nom, "Nominated (3)", "nominated_sae", HL, "o")]
    for yy, lab, key, col, mk in ref:
        row = cf.loc[key]
        c.plot([row.holdout_r_lo, row.holdout_r_hi], [yy, yy], color=col, lw=1.4,
               solid_capstyle="round", zorder=3)
        c.scatter([row.holdout_r], [yy], color=col, s=24, marker=mk, zorder=4)
        c.text(row.holdout_r_lo - 0.016, yy, lab, va="center", ha="right",
               fontsize=6.3, color=col)

    model_r = float(cf.loc["BrainLM frozen embedding (the model)", "holdout_r"])
    c.axvline(model_r, color="0.5", lw=0.7, ls="--", zorder=1)
    c.axvline(0, color="k", lw=0.6, zorder=2)
    c.set_yticks([])
    c.set_xlim(-0.16, 0.40)
    c.set_xticks([0.0, 0.1, 0.2, 0.3])
    c.set_xlabel("held-out $r$ (280 subjects)")
    c.set_title("(c) What generalizes", loc="left")
    # Two vertical lines would read as two axes; the meaningful one is r = 0.
    c.spines["left"].set_visible(False)

    for ax in (a, b, c):
        ax.set_ylim(*ylim)
    # The rule sits in (c) only: it separates feature sets from the two
    # reference points, and (a)/(b) have no reference points to separate.
    c.axhline(y_rule, color="0.8", lw=0.6, zorder=1)

    # The space the reference block frees in (a) carries the colour key.
    a.text(0.03, y_model + 0.10, "timescale", color=HL, fontsize=7, va="center")
    a.text(0.03, y_nom + 0.10, "connectivity", color=COG, fontsize=7, va="center")

    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")
    print(d[["label", "encoding_r2", "pct_of_model", "holdout_r", "p"]]
          .to_string(index=False))


if __name__ == "__main__":
    import sys
    if "--split" in sys.argv and sys.argv[sys.argv.index("--split") + 1] == "train":
        make(OUT / "fig_story_train.pdf", split="train")
    else:
        make()
    print(f"\n-> {OUT}")
