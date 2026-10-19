#!/usr/bin/env python3
"""Network and statistic-family enrichment of the attributed SAE features.

Codifies the step between attribution and nomination, which was previously computed
by hand: take the smallest set of top-ranked features reaching half the prediction
variance (k_to_50pct, from sae_attribution_hcp.py), look each one up in the
target-independent characterisation (characterise_sae_hcp.py), and report

  network     how many fire predominantly in each network, against the share of the
              other characterised features that do (binomial test for the top one)
  family      the share of each statistic family among the features' best-matching
              signal statistic, divided by the same share among the rest

For PMAT24 this must reproduce the manuscript: 27 of 29 in the central executive
network against a 47% baseline, lag-1 autocorrelation enriched 2.7x and global
connectivity 3.0x.

The baseline above is the rest of the dictionary, which does not answer whether the
READOUT picks out these features. The attributed features are dense and vary a lot
between subjects, and any readout's top features are drawn from that group. So the
same top-k selection is repeated under random readouts (a Gaussian direction with the
fitted norm, and a permutation of the fitted weights, as in sae_attribution_hcp.py),
and the real counts are placed in that null.

Usage:
    python src/04_interpretability/sae_enrichment_hcp.py [--target CogTotalComp_Unadj]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "04_interpretability"))
from common import load_config, resolve  # noqa: E402
from sae_attribution_hcp import ridge, subject_means  # noqa: E402


def family(stat: str) -> str:
    return stat.split("_", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--target", default=None)
    ap.add_argument("--null-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    target = args.target or cfg["label_column"]
    sfx = "" if target == cfg["label_column"] else f"_{target}"
    sae = ROOT / "results" / args.dataset / "brainlm" / "interpretability" / "sae"

    feats = pd.read_csv(sae / f"sae_attribution_features{sfx}.csv")
    summ = pd.read_csv(sae / f"sae_attribution_summary{sfx}.csv").iloc[0]
    char = pd.read_csv(sae / "sae_feature_characterisation.csv")
    k = int(summ.k_to_50pct)

    ranked = feats.dropna(subset=["rank"]).sort_values("rank")
    top_ids = ranked.feature.head(k).astype(int)
    c = char.set_index("feature")
    top = c.loc[c.index.intersection(top_ids)]
    rest = c.drop(top.index)
    print(f"{target}: readout CV r {summ.readout_cv_r:+.3f}, R2 {summ.readout_cv_r2:+.4f}; "
          f"k_to_50pct = {k}, {len(top)} of them characterised, {len(rest)} other features")

    net_top = top.top_network.value_counts()
    net_base = rest.top_network.value_counts(normalize=True)
    lead = net_top.index[0]
    p = stats.binomtest(int(net_top.iloc[0]), len(top), float(net_base.get(lead, 0)),
                        alternative="greater").pvalue
    print(f"\nnetwork: {net_top.iloc[0]}/{len(top)} fire predominantly in {lead}, against a "
          f"{net_base.get(lead, 0):.0%} baseline (binomial p = {p:.1e})")
    print(net_top.to_string())

    fam_top = top.best_signal_stat.map(family).value_counts(normalize=True)
    fam_rest = rest.best_signal_stat.map(family).value_counts(normalize=True)
    enr = (fam_top / fam_rest).dropna().sort_values(ascending=False)
    print("\nstatistic family enrichment (share among top / share among rest):")
    print(enr.to_string(float_format=lambda v: f"{v:.2f}x"))

    # --- random-readout null -------------------------------------------------------
    ckpt = torch.load(sae / "sae.pt", map_location="cpu", weights_only=False)
    D = ckpt["state_dict"]["dec.weight"].numpy().astype(np.float64)
    scale = float(ckpt["scale"])
    wm = np.load(sae / "sae_window_means.npz", allow_pickle=True)
    subj_w = np.asarray(wm["subject"]).astype(str)
    split = ROOT / "results" / args.dataset / "splits" / "train_subjects.txt"
    train_ids = {l.strip() for l in open(split) if l.strip()}
    pheno = pd.read_csv(resolve(cfg["metadata"]["phenotype"]), low_memory=False)
    pheno["Subject"] = pheno["Subject"].astype(str)
    tgt = pheno.set_index("Subject")[target]
    keep_w = np.array([s in train_ids and pd.notna(tgt.get(s, np.nan)) for s in subj_w])
    subs = sorted(set(subj_w[keep_w]))
    sidx = {s: i for i, s in enumerate(subs)}
    subj_idx = np.array([sidx[s] for s in subj_w[keep_w]])
    Zbar_s = subject_means(np.asarray(wm["Zbar"])[keep_w], subj_idx, len(subs))
    Emean_w = np.asarray(wm["emb_mean"]).astype(np.float64)[keep_w]
    y = np.array([float(tgt[s]) for s in subs])
    y = (y - y.min()) / (y.max() - y.min())
    pipe = ridge().fit(Emean_w, y[subj_idx])
    w_eff = pipe[-1].coef_ / pipe[0].scale_
    char_net = c.top_network.reindex(range(D.shape[1]))
    char_fam = c.best_signal_stat.reindex(range(D.shape[1])).map(
        lambda v: family(v) if isinstance(v, str) else None)

    def top_k(w):
        sv = w @ D
        sd = (scale * Zbar_s * sv[None, :]).std(0)
        alive = (Zbar_s > 0).any(0) & (np.abs(sv) > 0)
        sd[~alive] = -np.inf
        return np.argsort(-sd)[:k]

    def stats_of(ids):
        net = char_net.iloc[ids].dropna()
        fam = char_fam.iloc[ids].dropna()
        return {"n_lead": int((net == lead).sum()), "n_char": len(net),
                "ac1": float((fam == "ac1").mean()), "fc": float((fam == "fc").mean())}

    real = stats_of(top_k(w_eff))
    rng = np.random.default_rng(args.seed)
    null_rows = []
    for kind in ("gaussian", "permuted"):
        for _ in range(args.null_trials):
            if kind == "gaussian":
                g = rng.standard_normal(w_eff.shape)
                wr = g / np.linalg.norm(g) * np.linalg.norm(w_eff)
            else:
                wr = rng.permutation(w_eff)
            null_rows.append({"kind": kind, **stats_of(top_k(wr))})
    null = pd.DataFrame(null_rows)
    print(f"\nrandom-readout null ({args.null_trials} per kind), top {k}:")
    null_summary = []
    for kind, g in null.groupby("kind"):
        row = {"kind": kind}
        for m in ("n_lead", "ac1", "fc"):
            pv = (1 + (g[m] >= real[m]).sum()) / (1 + len(g))
            row[f"{m}_null_mean"], row[f"{m}_real"], row[f"{m}_p"] = g[m].mean(), real[m], pv
        null_summary.append(row)
        print(f"  {kind:<9} {lead}: real {real['n_lead']}/{real['n_char']} vs null mean "
              f"{g.n_lead.mean():.1f} (p {row['n_lead_p']:.3f}) | ac1 share real "
              f"{real['ac1']:.2f} vs {g.ac1.mean():.2f} (p {row['ac1_p']:.3f}) | global fc "
              f"real {real['fc']:.2f} vs {g.fc.mean():.2f} (p {row['fc_p']:.3f})")
    pd.DataFrame(null_summary).assign(target=target, k=k).to_csv(
        sae / f"sae_enrichment_null{sfx}.csv", index=False)

    out = pd.DataFrame({"family": enr.index, "enrichment": enr.values})
    out["target"], out["k"], out["lead_network"] = target, k, lead
    out["lead_network_count"], out["lead_network_baseline"] = int(net_top.iloc[0]), float(net_base.get(lead, 0))
    out["binomial_p"] = p
    out.to_csv(sae / f"sae_enrichment{sfx}.csv", index=False)
    print(f"\n-> {sae}/sae_enrichment{sfx}.csv")


if __name__ == "__main__":
    main()
