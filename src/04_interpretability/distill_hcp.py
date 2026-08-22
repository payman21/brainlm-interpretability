#!/usr/bin/env python3
"""Plan section 7.3 -- distil the model into explicit features.

The Goodfire payoff step. Section 7.0's probe showed the frozen embedding encodes
connectivity and temporal-dependence structure strongly (global FC R2 ~0.78, lag-1
autocorrelation ~0.76, network FC up to ~0.72) while predicting the phenotype only
weakly. If a handful of EXPLICIT features computed from the raw parcel timeseries
recovers the model's predictive performance, then those features are the mechanism,
stated in terms a domain expert can test.

Two questions, both answered here:

  target fidelity   can explicit features predict the phenotype as well as the
                    frozen embedding does?  (model: R2 0.075, r 0.284)
  model fidelity    can explicit features predict THE MODEL'S OWN PREDICTIONS?
                    This is the stronger test: it measures how much of the model's
                    function the simple features reproduce, regardless of how well
                    either predicts the target.

Feature sets are nested so the marginal value of each family is visible, rather
than reporting one lumped number.

Usage:
    python src/04_interpretability/distill_hcp.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
from common import load_config, resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402

# The five scale-invariant signal families, and nothing else. This has to be named
# rather than written as `lambda c: True`, because signal_features_per_window.csv.gz
# also carries the probe's deliberately-unlearnable columns -- absolute amplitude
# (var_, rawvar_, rawalff_, iqr_) and provenance identity (window_index,
# runphase_, session_). Those exist to be probed and FAIL; sweeping them into
# "all explicit" would distil the model onto preprocessing artefacts.
SIGNAL = ("fc_", "netfc_", "ac1_", "slope_", "falff_")

SETS = {
    "fc_global":        lambda c: c.startswith("fc_"),
    "netfc":            lambda c: c.startswith("netfc_"),
    "ac1":              lambda c: c.startswith("ac1_"),
    "spectral":         lambda c: c.startswith("slope_") or c.startswith("falff_"),
    "fc_global+ac1":    lambda c: c.startswith("fc_") or c.startswith("ac1_"),
    "netfc+ac1":        lambda c: c.startswith("netfc_") or c.startswith("ac1_"),
    "all_explicit":     lambda c: c.startswith(SIGNAL),
}


def r2(y, p):
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--target", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    target = args.target or cfg["label_column"]
    interp = ROOT / f"results/{args.dataset}/brainlm/interpretability"

    feat = pd.read_csv(interp / "signal_features_per_window.csv.gz")
    # Subject ids round-trip through CSV as int64; the phenotype index is str,
    # so without this cast the join silently matches nothing.
    subj_col = feat.pop("subject").astype(str)
    # Subject-level features: average each window feature within subject, which is
    # the same aggregation the winning frozen readout used on the embeddings.
    F = feat.groupby(subj_col.values).mean()
    print(f"explicit features: {F.shape[0]} subjects x {F.shape[1]} features")

    pheno = pd.read_csv(resolve(cfg["metadata"]["phenotype"]), low_memory=False)
    pheno["Subject"] = pheno["Subject"].astype(str)
    tgt = pheno.set_index("Subject")[target]
    subs = [s for s in F.index if pd.notna(tgt.get(s, np.nan))]
    F = F.loc[subs]
    y_raw = np.array([tgt[s] for s in subs], float)
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

    # The model's own subject-level predictions, from the frozen embedding.
    z = np.load(ROOT / f"results/{args.dataset}/brainlm/embeddings/frozen_embeddings.npz",
                allow_pickle=True)
    sid = z["subject_id"].astype(str)
    m = np.isin(sid, subs)
    order = {s: i for i, s in enumerate(subs)}
    si = np.array([order[s] for s in sid[m]])
    E = np.stack([z["mean"][m][si == i].mean(0) for i in range(len(subs))]).astype(np.float64)

    kf = KFold(args.folds, shuffle=True, random_state=args.seed)
    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 6, 25)))

    model_pred = cross_val_predict(ridge(), E, y, cv=kf)
    print(f"\nfrozen embedding -> {target}: R2 {r2(y, model_pred):+.4f}  "
          f"r {stats.pearsonr(y, model_pred)[0]:+.4f}   [the number to recover]")

    rows = []
    for name, sel in SETS.items():
        cols = [c for c in F.columns if sel(c)]
        if not cols:
            continue
        X = F[cols].values.astype(float)
        X = np.nan_to_num(X, nan=np.nanmean(X))
        p_t = cross_val_predict(ridge(), X, y, cv=kf)
        p_m = cross_val_predict(ridge(), X, model_pred, cv=kf)
        rows.append({
            "feature_set": name, "n_features": len(cols),
            "target_r2": r2(y, p_t), "target_r": float(stats.pearsonr(y, p_t)[0]),
            "model_fidelity_r2": r2(model_pred, p_m),
            "model_fidelity_r": float(stats.pearsonr(model_pred, p_m)[0]),
        })
        print(f"  {name:<16} n={len(cols):>3}  target R2 {rows[-1]['target_r2']:+.4f} "
              f"r {rows[-1]['target_r']:+.4f}  |  model-fidelity R2 "
              f"{rows[-1]['model_fidelity_r2']:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    base = r2(y, model_pred)
    df["frac_of_model"] = df["target_r2"] / base if base > 0 else np.nan
    print("\n=== distillation ===")
    print(df.to_string(index=False))
    print(f"\nfrac_of_model = explicit-feature R2 / frozen-embedding R2 ({base:+.4f})")

    # --- per-feature and per-network outputs -------------------------------------
    # Two tables the manuscript needs and that were previously only printed: the
    # single-feature ranking, and the per-network timescale localisation behind
    # @tbl:networks. Written to CSV so every number traces to a committed result file.
    singles = []
    for c in F.columns:
        if not c.startswith(SIGNAL):
            continue
        X = np.nan_to_num(F[[c]].values.astype(float))
        p_t = cross_val_predict(ridge(), X, y, cv=kf)
        singles.append({"feature": c, "r2": r2(y, p_t),
                        "frac_of_model": r2(y, p_t) / base if base > 0 else np.nan})
    sdf = pd.DataFrame(singles).sort_values("r2", ascending=False)
    print("\n=== single features, top 8 ===")
    print(sdf.head(8).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # Per-network timescale: the three measurements of slow-versus-fast fluctuation
    # for one network, together. This localises the mechanism across the AA-7 networks.
    nets = sorted({c.split("_", 1)[1] for c in F.columns if c.startswith("ac1_")})
    net_rows = []
    for n in nets:
        cols = [f"{p}_{n}" for p in ("ac1", "slope", "falff") if f"{p}_{n}" in F.columns]
        if not cols:
            continue
        X = np.nan_to_num(F[cols].values.astype(float))
        p_t = cross_val_predict(ridge(), X, y, cv=kf)
        net_rows.append({"network": n, "n_features": len(cols), "timescale_r2": r2(y, p_t),
                         "frac_of_model": r2(y, p_t) / base if base > 0 else np.nan})
    ndf = pd.DataFrame(net_rows).sort_values("timescale_r2", ascending=False)

    print("\n=== per-network timescale ===")
    print(ndf.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    out = ROOT / "results" / args.dataset / "brainlm" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"distillation_{target}.csv", index=False)
    sdf.to_csv(out / f"distillation_single_features_{target}.csv", index=False)
    ndf.to_csv(out / f"distillation_network_timescale_{target}.csv", index=False)
    stamp(out, "src/04_interpretability/distill_hcp.py", ROOT, target=target,
          n_subjects=len(subs), model_r2=base,
          best_set=str(df.loc[df.target_r2.idxmax(), "feature_set"]),
          best_target_r2=float(df.target_r2.max()),
          best_model_fidelity=float(df.model_fidelity_r2.max()))
    print(f"\n-> {out}/distillation_{target}.csv")


if __name__ == "__main__":
    main()
