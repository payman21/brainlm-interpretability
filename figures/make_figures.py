#!/usr/bin/env python3
"""Generate every figure in the HCP-YA manuscript from results/hcp_ya/.

Figures are generated, never hand-edited (paper/hcp_ya/README.md rule 3). Each
panel reads a committed CSV under results/hcp_ya/ so that any number in the paper
traces back to the run that produced it.

Usage:
    python paper/hcp_ya/figures/make_figures.py
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

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

COG = "#1f5f8b"
AFF = "#b0b0b0"
HL = "#c1440e"


def fig1_accessibility() -> None:
    """Nineteen targets, ordered, coloured by domain, with both controls."""
    d = pd.read_csv(RES / "metrics" / "frozen_controls_mean.csv")
    cont = d[d.kind == "continuous"].copy().sort_values("r2")
    pretty = {
        "PMAT24_A_CR": "Fluid intelligence", "CogCrystalComp_Unadj": "Crystallised cognition",
        "ReadEng_Unadj": "Reading (pronunciation)", "CogTotalComp_Unadj": "Total cognition",
        "CogFluidComp_Unadj": "Fluid cognition (composite)", "ListSort_Unadj": "List sorting",
        "PicVocab_Unadj": "Picture vocabulary", "ProcSpeed_Unadj": "Processing speed",
        "PicSeq_Unadj": "Picture sequence", "NEOFAC_N": "Neuroticism",
        "NEOFAC_E": "Extraversion", "NEOFAC_O": "Openness", "NEOFAC_A": "Agreeableness",
        "NEOFAC_C": "Conscientiousness", "PSQI_Score": "Sleep quality (PSQI)",
        "LifeSatisf_Unadj": "Life satisfaction", "PercStress_Unadj": "Perceived stress",
        "FearAffect_Unadj": "Fear/affect", "Sadness_Unadj": "Sadness",
    }
    cognitive = {"PMAT24_A_CR", "CogCrystalComp_Unadj", "ReadEng_Unadj", "CogTotalComp_Unadj",
                 "CogFluidComp_Unadj", "ListSort_Unadj", "PicVocab_Unadj",
                 "ProcSpeed_Unadj", "PicSeq_Unadj"}
    names = [pretty.get(t, t) for t in cont.target]
    colors = [COG if t in cognitive else AFF for t in cont.target]

    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    y = np.arange(len(cont))
    ax.barh(y, cont.r2, color=colors, height=0.72)
    ax.set_yticks(y, names)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("out-of-fold $R^2$ (frozen embedding $\\to$ target)")
    ax.set_title("Cognition is encoded; affect is not", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COG),
               plt.Rectangle((0, 0), 1, 1, color=AFF)]
    ax.legend(handles, ["cognitive", "affective / personality / sleep"],
              loc="lower right", frameon=False)
    fig.savefig(OUT / "fig1_accessibility.pdf")
    plt.close(fig)

    ctl = d[d.kind != "continuous"]
    print("  controls:", {r.target: round(r.score, 3) for r in ctl.itertuples()})


def fig3_encoding_vs_use() -> None:
    """The positive result: what the model encodes vs what it uses."""
    pr = pd.read_csv(INT / "probe_signal_features.csv")
    # The probe now reports every target under four embedding sources. Panel (a) is
    # about the pretrained model, so it must select one; without this the groupby
    # would average the pretrained score together with its own untrained baseline.
    pr = pr[pr.source == "frozen_mean"]
    fam = (pr.groupby("family").r2.mean()
             .rename(index={"fc": "Global FC", "ac1": "Autocorrelation",
                            "slope": "Spectral slope", "falff": "fALFF",
                            "netfc": "Network FC"}))
    order = ["Global FC", "Autocorrelation", "Spectral slope", "fALFF", "Network FC"]
    di = pd.read_csv(RES / "metrics" / "distillation_PMAT24_A_CR.csv")
    keep = {"fc_global": "Global FC (2)", "ac1": "Autocorrelation (7)",
            "spectral": "Spectral (14)", "netfc": "Network FC (28)",
            "all_explicit": "All explicit (51)"}
    di = di[di.feature_set.isin(keep)].copy()
    di["label"] = di.feature_set.map(keep)
    di = di.sort_values("frac_of_model")

    fig, (a, b) = plt.subplots(1, 2, figsize=(5.6, 2.5))
    ya = np.arange(len(order))
    a.barh(ya, [fam[o] for o in order],
           color=[HL if o in ("Autocorrelation", "Spectral slope") else COG for o in order],
           height=0.7)
    a.set_yticks(ya, order)
    a.set_xlim(0, 1)
    a.set_xlabel("out-of-fold $R^2$")
    a.set_title("(a) What the representation encodes", loc="left")

    yb = np.arange(len(di))
    cols = [HL if "Autocorrelation" in l or "Spectral" in l else COG for l in di.label]
    b.barh(yb, di.frac_of_model * 100, color=cols, height=0.7)
    b.set_yticks(yb, di.label)
    b.set_xlim(0, 100)
    b.set_xlabel("% of the model's predictive performance")
    b.set_title("(b) What the model uses", loc="left")
    for i, v in enumerate(di.frac_of_model * 100):
        b.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=6.5)
    fig.savefig(OUT / "fig3_encoding_vs_use.pdf")
    plt.close(fig)
    print("  encoding vs use written")


def fig2_probe_controls() -> None:
    """The probe read against its baselines, and against targets that should fail.

    This is the whole point of the controls: a column of high scores means nothing
    until the same probe is shown returning a low one. The lower block holds targets
    preprocessing removed before the model saw them, so they are the near-zero anchor.
    The token-identity probe that used to sit here as panel (b) is now figA1, beside
    its table in the appendix.
    """
    pr = pd.read_csv(INT / "probe_signal_features.csv")
    LAB = {"fc": "Global FC", "ac1": "Autocorrelation", "slope": "Spectral slope",
           "falff": "fALFF", "netfc": "Network FC", "var": "Amplitude (scaled)",
           "rawvar": "Raw parcel SD", "rawalff": "Raw ALFF", "iqr": "Parcel IQR",
           "window": "Window index", "runphase": "Phase encoding", "session": "Session"}
    VISIBLE = ["fc", "ac1", "slope", "falff", "netfc", "var"]
    REMOVED = ["rawvar", "rawalff", "iqr", "window", "runphase", "session"]
    fams = [f for f in VISIBLE + REMOVED if f in set(pr.family)]
    srcs = [("frozen_mean", "pretrained BrainLM", COG),
            ("random_init_mean", "randomly initialised", HL),
            ("random_proj", "random linear projection", AFF)]
    srcs = [s for s in srcs if s[0] in set(pr.source)]

    fig, a = plt.subplots(figsize=(4.4, 3.0))
    y = np.arange(len(fams))[::-1]
    h = 0.8 / len(srcs)
    for k, (src, lab, col) in enumerate(srcs):
        m = pr[pr.source == src].groupby("family").r2.mean()
        a.barh(y + (len(srcs) - 1) / 2 * h - k * h,
               [m.get(f, np.nan) for f in fams], height=h, color=col, label=lab)
    # Only the rule, no in-plot label: any text long enough to explain the split needs
    # a backing box, and the box clips the bar immediately below it. What the two blocks
    # are is stated in the caption instead, where it costs no data.
    a.axhline(y[len(VISIBLE) - 1] - 0.5, color="0.35", lw=0.8, ls=":")
    a.set_yticks(y, [LAB.get(f, f) for f in fams])
    a.set_xlim(0, 1.0)
    a.axvline(0, color="0.6", lw=0.6)
    a.set_xlabel("out-of-fold $R^2$")
    a.legend(frameon=False, loc="lower right", fontsize=6.5)
    fig.savefig(OUT / "fig2_probe_controls.pdf")
    plt.close(fig)
    print("  probe controls written")


def figA1_token_identity() -> None:
    """Token-level identity against the untrained twin, for the appendix.

    Parcel and patch position are handed to the encoder as additive input embeddings,
    so the pretrained-vs-untrained gap, not the height, is what is about pretraining.
    Split out of the probe-controls figure so that it sits beside its own table.
    """
    idf = []
    for tag, lab, col in [("frozen", "pretrained", COG),
                          ("random_init", "randomly initialised", HL)]:
        f = INT / f"probe_identity_tokens_{tag}.csv"
        if f.exists():
            idf.append((pd.read_csv(f), lab, col))
    if not idf:
        print("  token identity skipped (no input)")
        return

    keep = ["parcel_identity", "network_identity", "patch_position"]
    NAME = {"parcel_identity": "Parcel (424-way)", "network_identity": "Network (7-way)",
            "patch_position": "Patch position (10-way)"}
    fig, b = plt.subplots(figsize=(3.4, 2.2))
    yb = np.arange(len(keep))[::-1]
    hb = 0.8 / len(idf)
    for k, (d, lab, col) in enumerate(idf):
        d = d.set_index("target")
        b.barh(yb + (len(idf) - 1) / 2 * hb - k * hb,
               [d.loc[t, "value"] for t in keep], height=hb, color=col, label=lab)
    d0 = idf[0][0].set_index("target")
    for i, t in enumerate(keep):
        b.plot([d0.loc[t, "chance"]] * 2, [yb[i] - 0.4, yb[i] + 0.4],
               color="0.25", lw=1.0)
    b.set_yticks(yb, [NAME[t] for t in keep])
    # Headroom on the right so the legend does not sit on top of the bars: the
    # patch-position bars run to 1.00 and would otherwise pass under it.
    b.set_xlim(0, 1.34)
    b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    b.set_xlabel("top-1 accuracy (tick = chance)")
    b.legend(frameon=False, loc="upper right", fontsize=6.5)
    fig.savefig(OUT / "figA1_token_identity.pdf")
    plt.close(fig)
    print("  token identity written")


def fig4_sae_attribution() -> None:
    """What the model's prediction relies on: the ablation curve, and the 29 features.

    Panel (a) plots the whole curve rather than the reported point, because k=29 was
    fixed by a rule written down after k=9 failed and both have to stay visible. The
    random readouts were evaluated at the reported k only, so they are drawn as the two
    markers they are and not as a band across k we never computed.

    Panel (b) is why reading this dictionary as a null was wrong. The features carrying
    the prediction are a ~1% minority; in an aggregate summary of all 2,637 characterised
    features they disappear into the mass at zero.
    """
    cum = pd.read_csv(INT / "sae" / "sae_attribution_cumulative.csv")
    ctl = pd.read_csv(INT / "sae" / "sae_attribution_controls.csv")
    smy = pd.read_csv(INT / "sae" / "sae_attribution_summary.csv").iloc[0]
    ch = pd.read_csv(INT / "sae" / "sae_feature_characterisation.csv")
    feat = pd.read_csv(INT / "sae" / "sae_attribution_features.csv")
    k_top = int(smy.top_k)

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.2, 2.6))

    a.axhline(0, color="0.6", lw=0.6)
    a.axhline(0.5, color="0.8", lw=0.7, ls=":")
    a.plot(cum.k, cum.ablation_share, color=COG, lw=1.5, zorder=3,
           label="top-$k$ features")
    for kk, dx, dy in [(9, 4, -10), (k_top, 5, -2)]:
        v = float(cum.loc[cum.k == kk, "ablation_share"].iloc[0])
        a.scatter([kk], [v], color=COG, s=20, zorder=5)
        a.annotate(f"$k={kk}$: {v * 100:.0f}%", (kk, v), textcoords="offset points",
                   xytext=(dx, dy), fontsize=6.5, color=COG)
    marks = [("random readout (gaussian)", "o", "random readout, Gaussian"),
             ("random readout (permuted)", "s", "random readout, permuted")]
    for name, mk, lab in marks:
        row = ctl[ctl.control == name]
        if len(row):
            a.scatter([k_top], [float(row.mean_share_at_top.iloc[0])], color=HL,
                      marker=mk, s=26, zorder=6, label=lab)
    a.set_xscale("log")
    a.set_xlabel("features retained, $k$ (log scale)")
    a.set_ylabel("share of prediction variance")
    a.set_title("(a) Fixed-weight ablation", loc="left")
    a.legend(frameon=False, loc="lower right", fontsize=6.5)

    is_top = ch.feature.isin(set(feat.loc[feat["rank"] <= k_top, "feature"]))
    bins = np.linspace(-0.06, 0.45, 55)
    b.hist(ch.loc[~is_top, "signal_r2"], bins=bins, color=AFF,
           label=f"other {int((~is_top).sum()):,} features")
    b.hist(ch.loc[is_top, "signal_r2"], bins=bins, color=HL,
           label=f"the {int(is_top.sum())} attributed")
    b.set_yscale("log")
    b.axvline(0, color="0.6", lw=0.6)
    b.set_xlabel("variance explained by signal statistics ($R^2$)")
    b.set_ylabel("features (log scale)")
    b.set_title("(b) The attributed features are nameable", loc="left")
    b.legend(frameon=False, fontsize=6.5)

    fig.savefig(OUT / "fig4_sae_attribution.pdf")
    plt.close(fig)
    print(f"  k={k_top} share {smy.share_at_top_k:.3f}, "
          f"fidelity R2 {smy.fidelity_r2:.3f} (the ceiling on that share)")


def fig5_holdout_forest() -> None:
    """The held-out test: r with bootstrap intervals, one evaluation on 280 subjects.

    Every declared feature set appears, including the ones that read badly, because the
    design of the test is that nothing may be dropped after seeing it.
    """
    d = pd.read_csv(RES / "metrics" / "distill_confirm_PMAT24_A_CR.csv")
    MODEL = "BrainLM frozen embedding (the model)"
    LAB = {MODEL: "BrainLM frozen embedding (512)",
           "nominated_sae": "Nominated by attribution (3)",
           "spectral": "Spectral: slope + fALFF (14)",
           "ac1": "Lag-1 autocorrelation (7)",
           "centexe_only": "Central executive only (10)",
           "all_explicit": "All explicit (51)",
           "netfc": "Network FC (28)",
           "fc_global": "Global FC (2)",
           "bootstrap_90pct": "Bootstrap-selected (1)"}
    d = d[d.feature_set.isin(LAB)].copy()
    d["label"] = d.feature_set.map(LAB)
    d = d.sort_values("holdout_r").reset_index(drop=True)

    def colour(row) -> str:
        if row.feature_set == MODEL:
            return "#333333"
        if row.feature_set == "nominated_sae":
            return HL
        return COG if row.holdout_r_p < 0.05 else AFF

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for i, r in enumerate(d.itertuples()):
        c = colour(r)
        ax.plot([r.holdout_r_lo, r.holdout_r_hi], [i, i], color=c, lw=1.4,
                solid_capstyle="round", zorder=3)
        ax.scatter([r.holdout_r], [i], color=c, s=26, zorder=4,
                   marker="D" if r.feature_set == MODEL else "o")
    model_r = float(d.loc[d.feature_set == MODEL, "holdout_r"].iloc[0])
    ax.axvline(model_r, color="0.5", lw=0.7, ls="--", zorder=1)
    ax.axvline(0, color="k", lw=0.6, zorder=2)
    ax.set_yticks(np.arange(len(d)), d.label)
    ax.set_xlabel("held-out $r$ with fluid intelligence (280 subjects)")
    ax.set_title("Evaluated once, on subjects held out of every step", loc="left")
    fig.savefig(OUT / "fig5_holdout_forest.pdf")
    plt.close(fig)
    nom = d[d.feature_set == "nominated_sae"].iloc[0]
    print(f"  model r {model_r:.3f} vs nominated r {nom.holdout_r:.3f} "
          f"[{nom.holdout_r_lo:.3f}, {nom.holdout_r_hi:.3f}]")


if __name__ == "__main__":
    # Numbered by display order in manuscript.md; figA* are appendix figures.
    for f in (fig1_accessibility, fig2_probe_controls, fig3_encoding_vs_use,
              fig4_sae_attribution, fig5_holdout_forest, figA1_token_identity):
        print(f"{f.__name__} ...")
        f()
    print(f"\n-> {OUT}")
