#!/usr/bin/env python3
"""Plan 7.4 step 2b -- do SAE features track a parcel's STATE, not its identity?

characterise_sae_hcp.py tested two families and both came back near-null for the
typical feature: median eta2_parcel 0.001, median window-level signal_r2 -0.004. But
the ceiling on that second test was itself tiny -- median eta2_window 0.014 -- so it
was never in a position to find much. Almost none of a feature's variance is
attributable to which window a token came from.

That is diagnostic, not disappointing. It says features vary WITHIN a window, across
parcels, without being tied to a fixed parcel. The obvious candidate is parcel state:
"this parcel is currently slow" rather than "this is parcel 137" or "this window is
slow overall". Every statistic in the previous test was window-global or averaged
over a whole network, so a parcel-state feature is invisible to it by construction.

This script computes each statistic per parcel per window and matches it to the token
that carries that parcel in that window. Both stable and fluctuating parts are kept:

    <stat>          the parcel's value in this window
    <stat>_dev      the same, minus that parcel's mean over all its windows

The `_dev` block is the one that matters. It has parcel identity removed by
construction, so any variance it explains cannot be anatomy in disguise -- which is
exactly the confound that sank gradient attribution here.

Usage:
    python src/04_interpretability/parcel_state_sae_hcp.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
from common import load_config, resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402

TR = 0.72
STATS = ["ac1", "slope", "falff", "fcstr"]


def parcel_stats(w: np.ndarray) -> np.ndarray:
    """Per-parcel statistics of one 424 x 200 window -> (424, 4)."""
    x = w - w.mean(1, keepdims=True)
    sd = x.std(1)
    ok = sd > 0
    xz = np.zeros_like(x)
    xz[ok] = x[ok] / sd[ok, None]

    ac1 = np.zeros(x.shape[0])
    ac1[ok] = (xz[ok, :-1] * xz[ok, 1:]).mean(1)

    fc = (xz @ xz.T) / x.shape[1]
    np.fill_diagonal(fc, 0.0)
    fcstr = fc.sum(1) / max(fc.shape[0] - 1, 1)

    f, pxx = sps.welch(x, fs=1.0 / TR, nperseg=min(128, x.shape[1]), axis=1)
    keep = f > 0
    lf = np.log10(f[keep])
    lp = np.log10(np.maximum(pxx[:, keep], 1e-20))
    lf_c = lf - lf.mean()
    slope = (lp - lp.mean(1, keepdims=True)) @ lf_c / (lf_c @ lf_c)
    band = (f >= 0.01) & (f <= 0.08)
    tot = pxx.sum(1)
    falff = np.divide(pxx[:, band].sum(1), tot, out=np.zeros_like(tot), where=tot > 0)
    return np.stack([ac1, slope, falff, fcstr], 1)


def oof_r2_blocked(X, Z, groups, block=256, alpha=1.0, folds=5):
    """Out-of-fold R2 from a few predictors to every column of a huge Z.

    Z is 499,000 x 2,540; materialising predictions for all of it in float64 is
    10 GB, so features are done in blocks and only the two sums of squares are kept.
    """
    n_f = Z.shape[1]
    ss_res = np.zeros(n_f)
    ss_tot = np.zeros(n_f)
    splits = list(GroupKFold(folds).split(X, groups=groups))
    for b0 in range(0, n_f, block):
        b1 = min(b0 + block, n_f)
        Yb = Z[:, b0:b1].astype(np.float64)
        gm = Yb.mean(0)
        ss_tot[b0:b1] = ((Yb - gm) ** 2).sum(0)
        for tr, te in splits:
            Xtr, Xte = X[tr], X[te]
            mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-12)
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            ym = Yb[tr].mean(0)
            G = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
            B = np.linalg.solve(G, Xtr.T @ (Yb[tr] - ym))
            ss_res[b0:b1] += ((Yb[te] - (Xte @ B + ym)) ** 2).sum(0)
        print(f"  features {b1}/{n_f}", flush=True)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--model", default="brainlm")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    interp = ROOT / "results" / args.dataset / args.model / "interpretability"
    d = np.load(interp / "sae" / "sae_features.npz", allow_pickle=True)
    Z, meta, alive = d["Z"], d["meta"], d["alive"]
    cols = list(d["meta_cols"])
    wrow = meta[:, cols.index("window_row")].astype(int)
    parcel = meta[:, cols.index("parcel")].astype(int)
    subject, run, window = d["subject"].astype(str), d["run"].astype(str), d["window"]

    ts_dir = resolve(cfg["processed_path"]) / "timeseries"
    n_time, n_par = 200, 424
    P = np.full((len(window), n_par, len(STATS)), np.nan)
    cache_key, cache = None, None
    for i, (sb, rn, wn) in enumerate(zip(subject, run, window)):
        if cache_key != (sb, rn):
            cache = np.load(ts_dir / f"{sb}_{rn}.npy").astype(np.float32)
            cache_key = (sb, rn)
        seg = cache[:, int(wn) * n_time:(int(wn) + 1) * n_time]
        if seg.shape[1] == n_time:
            P[i] = parcel_stats(seg)
        if (i + 1) % 500 == 0 or i + 1 == len(window):
            print(f"  parcel stats {i+1}/{len(window)}", flush=True)

    # Parcel-mean removal: what is left fluctuates around each parcel's own baseline,
    # so it cannot be explained by which parcel a token is.
    with np.errstate(invalid="ignore"):
        pmean = np.nanmean(P, axis=0)                      # (424, 4)
    D = P - pmean[None]

    keep = np.where(alive & ((Z > 0).sum(0) >= 50))[0]
    Zk = Z[:, keep]

    tok_state = P[wrow, parcel]        # (n_tokens, 4)
    tok_dev = D[wrow, parcel]
    good = np.isfinite(tok_state).all(1) & np.isfinite(tok_dev).all(1)
    print(f"\n{good.sum():,}/{len(good):,} tokens with finite parcel state | "
          f"{len(keep)} features")

    g = subject[wrow][good]
    Zg = Zk[good]
    print("\nridge from parcel state -> feature activation (out-of-fold, grouped by subject)")
    print(" [raw parcel state]")
    r2_state = oof_r2_blocked(tok_state[good], Zg, g)
    print(" [parcel-mean-removed state]")
    r2_dev = oof_r2_blocked(tok_dev[good], Zg, g)
    rng = np.random.default_rng(0)
    print(" [null: state rows permuted]")
    r2_null = oof_r2_blocked(tok_dev[good][rng.permutation(int(good.sum()))], Zg, g)

    # Which single statistic, and in which direction.
    Sz = tok_dev[good]
    Sz = (Sz - Sz.mean(0)) / np.maximum(Sz.std(0), 1e-12)
    out_r = np.zeros((len(STATS), len(keep)))
    for b0 in range(0, len(keep), 256):
        b1 = min(b0 + 256, len(keep))
        Yb = Zg[:, b0:b1].astype(np.float64)
        Yb = (Yb - Yb.mean(0)) / np.maximum(Yb.std(0), 1e-12)
        out_r[:, b0:b1] = (Sz.T @ Yb) / len(Sz)
    best_i = np.abs(out_r).argmax(0)
    best_r = out_r[best_i, np.arange(len(keep))]

    ch = pd.read_csv(interp / "sae" / "sae_feature_characterisation.csv")
    add = pd.DataFrame({
        "feature": keep,
        "state_r2": r2_state, "state_dev_r2": r2_dev, "state_dev_r2_null": r2_null,
        "best_state_stat": [STATS[i] for i in best_i], "best_state_r": best_r,
    })
    ch = ch.drop(columns=[c for c in add.columns if c != "feature" and c in ch.columns])
    ch = ch.merge(add, on="feature", how="left")

    print("\n=== parcel state (token level, directly comparable to eta2_parcel) ===")
    for c, lbl in [("state_r2", "raw parcel state"),
                   ("state_dev_r2", "parcel-mean removed"),
                   ("state_dev_r2_null", "  null (permuted)")]:
        print(f"  {lbl:<22} median {ch[c].median():+.4f}  mean {ch[c].mean():+.4f}  "
              f"max {ch[c].max():+.4f}")
    print(f"  eta2_parcel (identity)   median {ch.eta2_parcel.median():+.4f}  "
          f"mean {ch.eta2_parcel.mean():+.4f}  max {ch.eta2_parcel.max():+.4f}")
    for t in (0.05, 0.10, 0.20):
        print(f"  features with state_dev_r2 > {t:.2f}: "
              f"{int((ch.state_dev_r2 > t).sum())}/{len(ch)}  "
              f"(null: {int((ch.state_dev_r2_null > t).sum())})")
    print("\n  statistic each state-driven feature tracks (state_dev_r2 > 0.05):")
    sel = ch[ch.state_dev_r2 > 0.05]
    if len(sel):
        print(sel.best_state_stat.value_counts().to_string())
        print(f"\ntop {args.top} features by state_dev_r2:")
        print(sel.nlargest(args.top, "state_dev_r2")[
            ["feature", "state_dev_r2", "eta2_parcel", "signal_r2",
             "best_state_stat", "best_state_r", "top_network"]
        ].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # SEPARATE FILE, deliberately. This script used to write
    # sae_feature_characterisation.csv -- the same path characterise_sae_hcp.py writes,
    # and the file @tbl:features and fig4_sae_attribution.pdf both read. Running the two
    # in either order silently replaced one's output with the other's, with no error and
    # no way to tell from the file which had produced it. The parcel-state columns are an
    # extension of that characterisation, not a replacement for it, so they belong in
    # their own file and are joined on `feature` when needed.
    out = interp / "sae" / "sae_parcel_state.csv"
    ch.to_csv(out, index=False)
    stamp(interp / "sae", "src/04_interpretability/parcel_state_sae_hcp.py", ROOT,
          median_state_dev_r2=float(ch.state_dev_r2.median()),
          median_state_dev_r2_null=float(ch.state_dev_r2_null.median()),
          n_state_dev_gt_0p05=int((ch.state_dev_r2 > 0.05).sum()))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
