#!/usr/bin/env python3
"""Plan 7.4 step 2 -- what does each SAE feature respond to?

A sparse dictionary is only interpretable once you know what makes each feature
fire. This is the step that decides whether the SAE found anything worth having.

Each feature's activation across tokens is tested against two families:

  ANATOMY / POSITION   parcel identity, AA-7 network, temporal patch index
                       -- the architectural structure that already contaminated
                       gradient attribution. A dictionary that only encodes these
                       is a null result, and an informative one.

  SIGNAL STATISTICS    the window's lag-1 autocorrelation, spectral slope, fALFF
                       and network FC -- the vocabulary the probe showed is present,
                       and the family that carries the prediction.

The comparison is the point. If predictive features turn out to be signal-statistic
features, that is convergent evidence for the timescale result from a method that
knew nothing about the hand-crafted features. If they are anatomy/position features,
the SAE has rediscovered the architecture.

The two families live at different levels -- anatomy varies token to token, signal
statistics are constant within a window -- so they are put on one scale via
eta2_window: the share of a feature's TOKEN-level variance attributable to which
window the token came from. That is the ceiling on what any window-level property
can explain, and signal_r2 * eta2_window is the signal family's share of the same
token-level variance the anatomy eta-squareds are measured against.

Usage:
    python src/04_interpretability/characterise_sae_hcp.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "04_interpretability"))
from common import load_config, resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402
from probe_signal_features_hcp import window_features  # noqa: E402

RUNS = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]

NET7 = {1: "Visual", 2: "VentSal", 3: "CentExe", 4: "Sensorimotor",
        5: "DefaultMode", 6: "DorsSal", 7: "Subcortical"}


def eta_sq_all(Z: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Eta-squared of every feature at once, against one categorical grouping.

    The obvious implementation loops over groups, masking the full token array
    each time, per feature. At 424 parcels x 3,400 features x 499,000 tokens that
    is ~7e11 element operations and takes hours. bincount does the same grouped
    sums in one pass per feature block, vectorised over features.

    Returns an array of length n_features.
    """
    n = Z.shape[0]
    cnt = np.bincount(groups, minlength=n_groups).astype(np.float64)
    cnt_safe = np.maximum(cnt, 1)
    gm = Z.mean(0, dtype=np.float64)
    ss_tot = ((Z.astype(np.float64) - gm) ** 2).sum(0)
    # grouped sums: (n_groups, n_features)
    sums = np.zeros((n_groups, Z.shape[1]), dtype=np.float64)
    np.add.at(sums, groups, Z.astype(np.float64))
    means = sums / cnt_safe[:, None]
    ss_bet = (cnt[:, None] * (means - gm) ** 2).sum(0)
    out = np.divide(ss_bet, ss_tot, out=np.zeros_like(ss_bet), where=ss_tot > 0)
    return out


def oof_r2_multi(X: np.ndarray, Y: np.ndarray, groups: np.ndarray,
                 alpha: float = 1.0, folds: int = 5) -> np.ndarray:
    """Out-of-fold R2 of a ridge from X to every column of Y, grouped by `groups`.

    Y has thousands of columns but X has 51, so this is one small solve per fold
    rather than one regression per feature.
    """
    P = np.zeros_like(Y)
    for tr, te in GroupKFold(folds).split(X, groups=groups):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.maximum(sd, 1e-12)
        ym = Y[tr].mean(0)
        m = Ridge(alpha=alpha, fit_intercept=False).fit((X[tr] - mu) / sd, Y[tr] - ym)
        P[te] = m.predict((X[te] - mu) / sd) + ym
    ss_res = ((Y - P) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
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
    Z, meta = d["Z"], d["meta"]
    cols = list(d["meta_cols"])
    wrow, parcel, net, patch = (meta[:, cols.index(c)] for c in
                                ["window_row", "parcel", "net7", "patch"])
    subject, run, window = d["subject"], d["run"], d["window"]

    # Per-window signal statistics for exactly the 5,000 sampled windows.
    #
    # signal_features_per_window.csv.gz carries only a `subject` column -- its run
    # and window identity is implicit in row order -- so joining to it means
    # reconstructing a positional key and hoping the enumeration matches. It does
    # not: that CSV covers the first 300 subjects, this sample covers 932. The
    # statistics are cheap and deterministic, so recompute them here against the
    # same windows the tokens actually came from. No join, nothing to get wrong.
    lab = pd.read_csv(resolve("external/atlases/A424/A424_Labels_AA-AAc_main_maps.csv"))
    lab.columns = [c.strip().lstrip("\ufeff") for c in lab.columns]
    net_name = lab.iloc[:, 5].values
    names = list(pd.unique(net_name))
    short = {n: n.split()[0][:4] + ("" if len(n.split()) == 1 else n.split()[1][:3])
             for n in names}
    net_of_parcel = np.array([names.index(n) for n in net_name])
    sn = [short[n] for n in names]

    ts_dir = resolve(cfg["processed_path"]) / "timeseries"
    n_time = 200
    rows, cache_key, cache = [], None, None
    for i, (sb, rn, wn) in enumerate(zip(subject.astype(str), run.astype(str), window), 1):
        if cache_key != (sb, rn):
            cache = np.load(ts_dir / f"{sb}_{rn}.npy").astype(np.float32)
            cache_key = (sb, rn)
        seg = cache[:, int(wn) * n_time:(int(wn) + 1) * n_time]
        # Runs shorter than 1200 TP do not fill all six windows. extract_token_acts
        # skipped those without removing them from the window list, so they are
        # present here with no tokens behind them. Keep a NaN row to hold the
        # position and drop it below, rather than shifting every window_row after it.
        rows.append(window_features(seg, net_of_parcel, len(names), sn)
                    if seg.shape[1] == n_time else {})
        if i % 500 == 0 or i == len(window):
            print(f"  signal features {i}/{len(window)}", flush=True)
    F = pd.DataFrame(rows)
    sig_cols = list(F.columns)
    S = F.values.astype(np.float64)
    print(f"\n{Z.shape[0]:,} tokens x {Z.shape[1]} features | "
          f"{S.shape[0]} windows x {S.shape[1]} signal statistics")

    alive = d["alive"]
    keep = np.where(alive & ((Z > 0).sum(0) >= 50))[0]
    Zk = Z[:, keep]
    print(f"characterising {len(keep)} features with >=50 activations "
          f"(matrix {Zk.shape}, {Zk.nbytes/1e9:.1f} GB)", flush=True)

    e_par = eta_sq_all(Zk, parcel.astype(int), int(parcel.max()) + 1)
    e_net = eta_sq_all(Zk, net.astype(int), int(net.max()) + 1)
    e_pat = eta_sq_all(Zk, patch.astype(int), int(patch.max()) + 1)
    e_win = eta_sq_all(Zk, wrow.astype(int), int(wrow.max()) + 1)

    # --- signal statistics, at the window level ---------------------------
    # A signal statistic is constant within a window, so it can only ever explain
    # the between-window part of a feature's variance -- that part is e_win, and it
    # is the honest ceiling to quote signal_r2 against. Aggregate each feature to a
    # per-window mean, then ridge from the 51 statistics to all features at once
    # (one 51x51 solve per fold, whatever the feature count) with folds grouped by
    # subject so no subject's windows straddle a split.
    n_win = int(wrow.max()) + 1
    wcnt = np.bincount(wrow.astype(int), minlength=n_win).astype(np.float64)
    W = np.zeros((n_win, Zk.shape[1]))
    np.add.at(W, wrow.astype(int), Zk.astype(np.float64))
    W /= np.maximum(wcnt, 1)[:, None]

    # Only windows that both carry tokens and yielded finite statistics.
    vw = (wcnt > 0) & np.isfinite(S).all(1)
    Sv, Wv, gv = S[vw], W[vw], subject.astype(str)[vw]
    print(f"window-level regression on {vw.sum()}/{n_win} usable windows, "
          f"{len(np.unique(gv))} subjects")

    sig_r2 = oof_r2_multi(Sv, Wv, gv)
    rng = np.random.default_rng(0)
    null_r2 = oof_r2_multi(Sv[rng.permutation(len(Sv))], Wv, gv)

    # Best single statistic per feature, by window-level correlation.
    Sz = (Sv - Sv.mean(0)) / np.maximum(Sv.std(0), 1e-12)
    Wz = (Wv - Wv.mean(0)) / np.maximum(Wv.std(0), 1e-12)
    R = (Sz.T @ Wz) / len(Sv)                     # (n_stats, n_features)
    best_i = np.abs(R).argmax(0)
    best_r = R[best_i, np.arange(R.shape[1])]

    # Where each feature fires most, computed with the same grouped-count trick.
    act = Zk > 0
    n_act = act.sum(0)
    net_counts = np.zeros((int(net.max()) + 1, len(keep)))
    np.add.at(net_counts, net.astype(int), act)
    pat_counts = np.zeros((int(patch.max()) + 1, len(keep)))
    np.add.at(pat_counts, patch.astype(int), act)
    top_net_i = net_counts.argmax(0)
    top_pat_i = pat_counts.argmax(0)

    df = pd.DataFrame({
        "feature": keep, "n_active": n_act,
        "frac_active": n_act / Zk.shape[0],
        "eta2_parcel": e_par, "eta2_network": e_net, "eta2_patch": e_pat,
        "eta2_window": e_win,
        "signal_r2": sig_r2, "signal_r2_null": null_r2,
        "signal_share_of_token_var": np.maximum(sig_r2, 0) * e_win,
        "best_signal_stat": [sig_cols[i] for i in best_i],
        "best_signal_r": best_r,
        "top_network": [NET7.get(int(i), "?") for i in top_net_i],
        "top_network_frac": net_counts[top_net_i, np.arange(len(keep))] / np.maximum(n_act, 1),
        "top_patch": top_pat_i,
        "top_patch_frac": pat_counts[top_pat_i, np.arange(len(keep))] / np.maximum(n_act, 1),
    })
    df["anatomy_position_score"] = df[["eta2_parcel", "eta2_network", "eta2_patch"]].max(axis=1)
    print(f"\ncharacterised {len(df)} active features")
    print("\nvariance explained by anatomy/position (eta^2, 0 = none, 1 = fully determined):")
    print(df[["eta2_parcel", "eta2_network", "eta2_patch"]]
          .describe().loc[["mean", "50%", "max"]].to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nfeatures whose activation is >50% explained by anatomy or position: "
          f"{int((df.anatomy_position_score > 0.5).sum())}/{len(df)}")
    print(f"                                          >20%: "
          f"{int((df.anatomy_position_score > 0.2).sum())}/{len(df)}")
    print(f"\nnetwork concentration (fraction of a feature's firings in its top network):")
    print(f"  median {df.top_network_frac.median():.3f}   "
          f"(chance ~0.26 for the largest network)")

    print("\n=== signal statistics (window level) ===")
    print(f"between-window share of token variance (eta2_window, the ceiling): "
          f"median {df.eta2_window.median():.3f}  mean {df.eta2_window.mean():.3f}")
    print(f"signal_r2 (51 statistics -> feature's per-window mean, out-of-fold): "
          f"median {df.signal_r2.median():.3f}  mean {df.signal_r2.mean():.3f}  "
          f"max {df.signal_r2.max():.3f}")
    print(f"  permuted-window null:                     "
          f"median {df.signal_r2_null.median():.3f}  mean {df.signal_r2_null.mean():.3f}  "
          f"max {df.signal_r2_null.max():.3f}")
    for t in (0.2, 0.5):
        print(f"features with signal_r2 > {t}: {int((df.signal_r2 > t).sum())}/{len(df)}"
              f"   (null: {int((df.signal_r2_null > t).sum())})")

    print("\nhead-to-head on ONE scale, share of token-level variance:")
    print(f"  anatomy/position  mean {df.anatomy_position_score.mean():.3f}   "
          f"median {df.anatomy_position_score.median():.3f}")
    print(f"  signal statistics mean {df.signal_share_of_token_var.mean():.3f}   "
          f"median {df.signal_share_of_token_var.median():.3f}")
    n_sig = int((df.signal_share_of_token_var > df.anatomy_position_score).sum())
    print(f"  features better explained by signal than by anatomy: {n_sig}/{len(df)}")

    print("\nwhich statistic each feature tracks best (features with signal_r2 > 0.2):")
    sel = df[df.signal_r2 > 0.2]
    if len(sel):
        fam = sel.best_signal_stat.str.replace(r"_.*", "", regex=True)
        print(fam.value_counts().to_string())
        print("\ntop features by signal_r2:")
        print(sel.nlargest(args.top, "signal_r2")[
            ["feature", "signal_r2", "eta2_window", "anatomy_position_score",
             "best_signal_stat", "best_signal_r", "top_network"]
        ].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out = interp / "sae"
    df.to_csv(out / "sae_feature_characterisation.csv", index=False)
    stamp(out, "src/04_interpretability/characterise_sae_hcp.py", ROOT,
          n_features=len(df),
          frac_anatomy_dominated=float((df.anatomy_position_score > 0.5).mean()),
          median_signal_r2=float(df.signal_r2.median()),
          median_signal_r2_null=float(df.signal_r2_null.median()),
          n_signal_r2_gt_0p2=int((df.signal_r2 > 0.2).sum()),
          frac_signal_beats_anatomy=float(
              (df.signal_share_of_token_var > df.anatomy_position_score).mean()))
    print(f"\n-> {out}/sae_feature_characterisation.csv")


if __name__ == "__main__":
    main()
