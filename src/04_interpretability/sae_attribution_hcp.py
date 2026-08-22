#!/usr/bin/env python3
"""Steps 3-5 -- which SAE features does the MODEL's prediction depend on?

This replaces the sparse readout in sae_readout_hcp.py, which fitted a NEW elastic net
from SAE features to the behavioural score. That measures what predicts the outcome.
The reference study measures what the existing classifier relies on, by pushing
gradients back through it. Those are different questions, and the difference is the
encoding-versus-use distinction this project makes everywhere else.

No gradients are needed here, because the readout being explained is linear.

The decomposition
-----------------
The readout that produces the headline result is `mean` / `window_then_mean` / ridge:
a ridge on the token-mean embedding, fitted over windows, predictions averaged per
subject. The SAE decomposes the very tokens that embedding is a mean of, and
mean-pooling is linear, so the decoder passes straight through it.

Writing a_t for a token's activation, d_i for decoder column i (unit norm), and
z_bar_i for feature i's mean activation over a window's 4,240 tokens:

    a_t     ~=  scale * ( sum_i z_i,t d_i  +  b_pre )  +  mu
    e_w      =  mean_t a_t  ~=  scale * ( sum_i z_bar_i,w d_i  +  b_pre )  +  mu
    y_hat_w  =  w . (e_w - m_s)/s_s + c        [ridge on a StandardScaler]

so with w_eff = w / s_s, feature i's contribution to subject s is

    c_i,s  =  scale * z_bar_i,s * ( w_eff . d_i )

where z_bar_i,s averages over all of that subject's windows -- which is exactly what
`window_then_mean` does, since averaging a linear function of the windows is the same as
applying it to their average. `mu`, `b_pre` and the intercept collapse into a constant
and drop out of any cross-subject comparison.

`w_eff . d_i` is one scalar per feature: 4,096 dot products, computed once.

What is reported
----------------
Features are ranked by sd of their contribution across subjects -- a feature with a
large but CONSTANT contribution moves the intercept and explains no between-subject
variance. Then the cumulative share, Var(sum of top k) / Var(sum of all), with the
value at k=9 called out so it sits against the reference study's "nine SAE features
responsible for the majority of classification performance".

Three checks, and each can fail
-------------------------------
  IDENTITY   The analytic contributions must sum to the pipeline's own prediction on the
             reconstructed embedding, to float precision. This is an assert, not a
             report: if it trips, the algebra above does not describe the readout.
  FIDELITY   How well the prediction from the SAE's RECONSTRUCTION tracks the prediction
             from the true embedding. Validation FVU was ~0.08 at token level, so this
             is not free. If fidelity is poor the decomposition is exact but explains a
             model that is not the model.
  CONTROLS   A random readout (both Gaussian and a permutation of the fitted weights)
             must not promote the same features -- the §2b control, moved into feature
             space, where it is a real test because w_eff . d_i genuinely depends on the
             fitted direction. And a readout-BLIND ranking by activation variance alone,
             to show the readout direction is doing work rather than the ranking falling
             out of which features simply vary most.

Everything runs on the TRAINING split only. The holdout is not read.

Run characterise_sae_hcp.py on the top-ranked features afterwards; do not reuse an
existing characterisation file, because feature indices are not stable across SAE runs.

Usage:
    python src/04_interpretability/sae_attribution_hcp.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
from common import load_config, resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402

ALPHAS = np.logspace(-2, 6, 25)


def ridge():
    """The readout being explained, identical to predict_frozen_hcp.py's `ridge`."""
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def subject_means(V: np.ndarray, subj_idx: np.ndarray, n_subj: int) -> np.ndarray:
    """Mean of V over each subject's rows. Equal weight per window, as window_then_mean."""
    out = np.zeros((n_subj, V.shape[1]), dtype=np.float64)
    np.add.at(out, subj_idx, V.astype(np.float64))
    cnt = np.bincount(subj_idx, minlength=n_subj).astype(np.float64)
    return out / np.maximum(cnt, 1)[:, None]


def cv_context_r2(Xw, yw, subj_idx, y_subj, n_subj, folds, seed):
    """Cross-validated window_then_mean R2 on the training split, for context only.

    Not used by the attribution. It exists so the reader can see that the model being
    decomposed performs comparably to the published 0.0748 rather than being some
    degenerate fit peculiar to this split.
    """
    oof = np.full(n_subj, np.nan)
    rng = np.random.default_rng(seed)
    fold_of_subj = rng.permutation(n_subj) % folds
    for f in range(folds):
        tr_m = fold_of_subj[subj_idx] != f
        te_m = ~tr_m
        if not te_m.any() or not tr_m.any():
            continue
        mdl = ridge().fit(Xw[tr_m], yw[tr_m])
        pw = mdl.predict(Xw[te_m])
        si = subj_idx[te_m]
        for s in np.unique(si):
            oof[s] = pw[si == s].mean()
    ok = ~np.isnan(oof)
    return r2(y_subj[ok], oof[ok]), float(stats.pearsonr(y_subj[ok], oof[ok])[0])


def default_ks(n: int) -> np.ndarray:
    return np.unique(np.clip(
        np.r_[np.arange(1, min(30, n) + 1), np.geomspace(1, max(n, 2), 40).astype(int)], 1, n))


def ablation_share(C: np.ndarray, order: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Fraction of the decomposed prediction's variance destroyed by ablating the top k.

    THE metric. Fixed weights: the prediction is exactly sum_i c_i + const, so removing
    a set S leaves sum_{i not in S} c_i, and the share is 1 - Var(residual)/Var(total).
    Nothing is refitted, so this asks what the features carry AS THE MODEL COMBINES THEM.

    Two wrong versions were tried first and both are instructive.
    Var(sum of top k)/Var(total) is not monotone and passed 1.27 at k=20 here, because
    contributions are signed and correlated. A projection R2 -- the prediction regressed
    on the k contribution columns -- is monotone and bounded, but it refits the
    coefficients, and that discards exactly the thing being measured: on this data a
    RANDOM readout's top 9 scored 0.747 against the real readout's 0.715, because almost
    any 9 varying directions can span a one-dimensional signal once you are allowed to
    reweight them. Only the fixed-weight version distinguishes a good selection from a
    bad one, which is what makes its control informative.
    """
    T = C.sum(1)
    denom = T.var()
    out = np.empty(len(ks))
    for j, k in enumerate(ks):
        resid = T - C[:, order[:k]].sum(1)
        out[j] = 1.0 - resid.var() / denom if denom > 0 else np.nan
    return out


def projection_r2(C: np.ndarray, order: np.ndarray, y: np.ndarray,
                  ks: np.ndarray) -> np.ndarray:
    """Share of `y`'s variance accounted for by the first k contribution columns.

    This is a PROJECTION R2 -- y regressed on the k columns -- not
    Var(sum of k columns) / Var(sum of all). The naive ratio is the obvious thing to
    write and it is wrong here: contributions are correlated and signed, so the
    variance of a partial sum is neither monotone in k nor bounded by the total. On
    this data it fell from 0.193 at k=1 to 0.086 at k=2 and passed 1.27 at k=20, which
    makes any single "top k accounts for X%" a cherry-picked point on a curve that
    wanders. The projection is monotone non-decreasing in k and lies in [0, 1].

    Returned in-sample. For k comparable to the number of subjects it approaches 1 by
    construction, so only small k should be quoted; `ks` keeps the evaluation to a grid
    rather than all 4,096 nested fits.
    """
    yc = y - y.mean()
    sst = float((yc ** 2).sum())
    out = np.empty(len(ks))
    for j, k in enumerate(ks):
        X = C[:, order[:k]]
        X = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[j] = 1.0 - float(((y - X @ beta) ** 2).sum()) / sst if sst > 0 else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--model", default="brainlm")
    ap.add_argument("--target", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-trials", type=int, default=20)
    ap.add_argument("--top", type=int, default=9,
                    help="the k called out in the summary; 9 mirrors the reference study")
    ap.add_argument("--sae-dir", type=Path, default=None,
                    help="directory holding sae.pt and sae_window_means.npz, and where the "
                         "attribution CSVs are written. Default is the canonical one")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    target = args.target or cfg["label_column"]
    interp = ROOT / "results" / args.dataset / args.model / "interpretability"
    sae_dir = args.sae_dir or (interp / "sae")

    ckpt = torch.load(sae_dir / "sae.pt", map_location="cpu", weights_only=False)
    D = ckpt["state_dict"]["dec.weight"].numpy().astype(np.float64)      # (512, width)
    b_pre = ckpt["state_dict"]["b_pre"].numpy().astype(np.float64)       # (512,)
    mu = np.asarray(ckpt["mu"], dtype=np.float64).reshape(1, -1)
    scale = float(ckpt["scale"])
    width = D.shape[1]

    wm = np.load(sae_dir / "sae_window_means.npz", allow_pickle=True)
    Zbar_w = np.asarray(wm["Zbar"])
    Emean_w = np.asarray(wm["emb_mean"]).astype(np.float64)
    subj_w = np.asarray(wm["subject"]).astype(str)
    if Zbar_w.shape[1] != width:
        raise SystemExit(
            f"sae_window_means.npz has width {Zbar_w.shape[1]} but sae.pt has {width}. "
            "These are from different SAE runs -- re-run encode_sae_windows_hcp.py.")

    split_dir = ROOT / "results" / args.dataset / "splits"
    if not (split_dir / "train_subjects.txt").exists():
        raise SystemExit(f"no split at {split_dir}. Run split_holdout_hcp.py first -- "
                         "attributing on all 938 subjects contaminates the holdout.")
    train_ids = {l.strip() for l in open(split_dir / "train_subjects.txt") if l.strip()}

    pheno = pd.read_csv(resolve(cfg["metadata"]["phenotype"]), low_memory=False)
    pheno["Subject"] = pheno["Subject"].astype(str)
    tgt = pheno.set_index("Subject")[target]

    # Training windows only, and only subjects with a target.
    keep_w = np.array([s in train_ids and pd.notna(tgt.get(s, np.nan)) for s in subj_w])
    subs = sorted(set(subj_w[keep_w]))
    sidx = {s: i for i, s in enumerate(subs)}
    subj_idx = np.array([sidx[s] for s in subj_w[keep_w]])
    n_subj = len(subs)
    if n_subj < args.folds * 2:
        raise SystemExit(
            f"only {n_subj} training subjects present in sae_window_means.npz. Either the "
            "encode step was run with --limit, or it covered subjects that are all in the "
            "holdout. Re-run encode_sae_windows_hcp.py over the full cohort.")
    Zbar_w, Emean_w = Zbar_w[keep_w], Emean_w[keep_w]
    print(f"training split: {n_subj} subjects, {keep_w.sum()} windows "
          f"({keep_w.sum()/n_subj:.1f} per subject), width {width}")

    # Min-max on the TRAINING targets only, following BrainLM section 3.4. R2 is
    # invariant to an affine transform of y, so this changes no reported number; it just
    # avoids taking the range from subjects the model is not allowed to see.
    y_raw = np.array([float(tgt[s]) for s in subs])
    lo, hi = y_raw.min(), y_raw.max()
    y_subj = (y_raw - lo) / (hi - lo)
    y_win = y_subj[subj_idx]

    # --- the readout being explained -------------------------------------------------
    pipe = ridge().fit(Emean_w, y_win)
    sc, rg = pipe[0], pipe[-1]
    w_eff = rg.coef_ / sc.scale_
    cv_r2, cv_r = cv_context_r2(Emean_w, y_win, subj_idx, y_subj, n_subj,
                                args.folds, args.seed)
    print(f"readout: ridge alpha {rg.alpha_:.4g} | in-split CV (window_then_mean) "
          f"R2 {cv_r2:+.4f} r {cv_r:+.4f}   [published full-cohort value: +0.0748]")

    # --- contributions ---------------------------------------------------------------
    Zbar_s = subject_means(Zbar_w, subj_idx, n_subj)          # (n_subj, width)
    E_s = subject_means(Emean_w, subj_idx, n_subj)            # (n_subj, 512)
    s_i = w_eff @ D                                           # (width,)
    C = scale * Zbar_s * s_i[None, :]                         # (n_subj, width)

    # IDENTITY CHECK. The reconstructed subject embedding, and the prediction from it,
    # must equal the analytic sum of contributions plus one constant.
    Ehat_s = scale * (Zbar_s @ D.T + b_pre) + mu
    yhat_recon = pipe.predict(Ehat_s)
    const0 = float(pipe.predict((scale * b_pre + mu.ravel()).reshape(1, -1))[0])
    yhat_from_C = C.sum(1) + const0
    err = np.abs(yhat_from_C - yhat_recon).max()
    # Tolerance scales with the size of the terms being summed, not with the size of the
    # result: individual contributions are large and cancel, so the float64 rounding floor
    # is set by sum|c_i|, which is orders of magnitude above the prediction itself.
    tol = 1e-8 * max(1.0, float(np.abs(C).sum(1).max()))
    assert err < tol, (
        f"decomposition does not reproduce the readout (max abs error {err:.3g} > {tol:.3g}). "
        "The algebra in this docstring does not describe the fitted pipeline.")
    print(f"identity check: max |sum_i c_i + const - predict(Ehat)| = {err:.3g}  OK")

    # FIDELITY. How much of the real model's prediction survives the dictionary.
    yhat_true = pipe.predict(E_s)
    fid_r2 = r2(yhat_true, yhat_recon)
    fid_r = float(stats.pearsonr(yhat_true, yhat_recon)[0])
    emb_fvu = float(((E_s - Ehat_s) ** 2).sum() / ((E_s - E_s.mean(0)) ** 2).sum())
    print(f"fidelity: prediction from SAE reconstruction vs from the true embedding, "
          f"R2 {fid_r2:+.4f} r {fid_r:+.4f} | pooled-embedding FVU {emb_fvu:.4f}")

    # --- ranking ---------------------------------------------------------------------
    alive = (Zbar_s > 0).any(0) & (np.abs(s_i) > 0)
    sd = C.std(0)
    order = np.argsort(-sd)
    order = order[alive[order]]
    # Two targets, and the difference between them is the whole caveat. Against
    # `yhat_recon` the decomposition is exact, so this is a clean statement about how
    # concentrated the dictionary's account is. Against `yhat_true` it is capped by
    # fidelity -- that is the share of the ACTUAL model's output, and it is the number
    # that matters.
    ks = default_ks(len(order))
    share = ablation_share(C, order, ks)              # the number to quote
    share_proj = projection_r2(C, order, yhat_recon, ks)   # refit upper bound, secondary
    at = lambda arr, k: float(arr[np.searchsorted(ks, k)]) if k in ks else float("nan")
    share_top, share_top_proj = at(share, args.top), at(share_proj, args.top)
    k50 = int(ks[np.argmax(share >= 0.50)]) if (share >= 0.50).any() else -1
    k90 = int(ks[np.argmax(share >= 0.90)]) if (share >= 0.90).any() else -1
    top = order[: args.top]
    var_amp = float(C[:, top].sum(1).var() / C[:, top].var(0).sum())
    print(f"\n{alive.sum()} live features | ablating the top {args.top} removes "
          f"{share_top:.1%} of the decomposed prediction's variance")
    print(f"  {k50} features reach 50% and {k90} reach 90% (-1 = never)")
    print(f"  refit upper bound at k={args.top} (projection R2, coefficients free): "
          f"{share_top_proj:.1%} -- reported only to show how much the freedom to "
          f"reweight flatters this")
    print(f"  covariance amplification over the top {args.top}: {var_amp:.2f}x "
          f"(>1 means they act together, <1 means they partly cancel)")

    # --- controls --------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    def top_set(sv: np.ndarray) -> set:
        c = scale * Zbar_s * sv[None, :]
        sdv = c.std(0)
        sdv[~alive] = -np.inf
        return set(np.argsort(-sdv)[: args.top].tolist())

    real = set(top.tolist())
    ctl = []
    for kind in ("gaussian", "permuted"):
        ov, sh = [], []
        for _ in range(args.control_trials):
            if kind == "gaussian":
                g = rng.standard_normal(w_eff.shape)
                wr = g / np.linalg.norm(g) * np.linalg.norm(w_eff)
            else:
                # Permuting the fitted weights keeps their marginal distribution and
                # norm, and destroys only which embedding dimension each belongs to.
                wr = rng.permutation(w_eff)
            sv = wr @ D
            ov.append(len(real & top_set(sv)) / args.top)
            # Rank with the RANDOM readout, then score with the REAL contributions.
            # That isolates selection quality: does a random direction pick features
            # that carry the actual model's prediction as well as the fitted one does?
            cv_ = scale * Zbar_s * sv[None, :]
            o = np.argsort(-cv_.std(0))
            o = o[alive[o]]
            sh.append(float(ablation_share(C, o, np.array([args.top]))[0])
                      if len(o) >= args.top else np.nan)
        ctl.append({"control": f"random readout ({kind})", "trials": args.control_trials,
                    "mean_top_overlap": float(np.mean(ov)), "max_top_overlap": float(np.max(ov)),
                    "mean_share_at_top": float(np.nanmean(sh))})
        print(f"  control, random readout ({kind}): top-{args.top} overlap with the real "
              f"set {np.mean(ov):.2f} mean / {np.max(ov):.2f} max over "
              f"{args.control_trials} draws")

    blind = np.argsort(-Zbar_s.std(0))
    blind = blind[alive[blind]][: args.top]
    ov_blind = len(real & set(blind.tolist())) / args.top
    ctl.append({"control": "readout-blind (activation variance)", "trials": 1,
                "mean_top_overlap": ov_blind, "max_top_overlap": ov_blind,
                "mean_share_at_top": float("nan")})
    print(f"  control, readout-blind ranking by activation variance: top-{args.top} "
          f"overlap {ov_blind:.2f}")

    # --- write ------------------------------------------------------------------------
    feats = pd.DataFrame({
        "feature": np.arange(width),
        "w_dot_decoder": s_i,
        "sd_contribution": sd,
        "mean_abs_contribution": np.abs(C).mean(0),
        "mean_activation": Zbar_s.mean(0),
        "frac_windows_nonzero": (Zbar_w > 0).mean(0),
        "alive": alive,
    })
    feats["rank"] = feats.feature.map({int(f): j + 1 for j, f in enumerate(order)})
    feats = feats.sort_values("rank", na_position="last")
    cum = pd.DataFrame({"k": ks, "ablation_share": share,
                        "projection_r2_refit_upper_bound": share_proj})
    summ = pd.DataFrame([{
        "target": target, "n_subjects_train": n_subj, "n_windows_train": int(keep_w.sum()),
        "width": width, "n_alive": int(alive.sum()),
        "readout_alpha": float(rg.alpha_), "readout_cv_r2": cv_r2, "readout_cv_r": cv_r,
        "identity_check_max_abs_err": float(err),
        "fidelity_r2": fid_r2, "fidelity_r": fid_r, "pooled_embedding_fvu": emb_fvu,
        "top_k": args.top, "share_at_top_k": share_top,
        "share_at_top_k_projection_refit": share_top_proj,
        "k_to_50pct": k50, "k_to_90pct": k90,
        "covariance_amplification_top_k": var_amp,
    }])

    print(f"\ntop {args.top} features:")
    print(feats.head(args.top)[["feature", "rank", "w_dot_decoder", "sd_contribution",
                                "mean_activation", "frac_windows_nonzero"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4g}"))

    feats.to_csv(sae_dir / "sae_attribution_features.csv", index=False)
    cum.to_csv(sae_dir / "sae_attribution_cumulative.csv", index=False)
    summ.to_csv(sae_dir / "sae_attribution_summary.csv", index=False)
    pd.DataFrame(ctl).to_csv(sae_dir / "sae_attribution_controls.csv", index=False)
    stamp(sae_dir, "src/04_interpretability/sae_attribution_hcp.py", ROOT,
          target=target, n_subjects_train=n_subj, width=width,
          fidelity_r2=fid_r2, share_at_top_k=share_top, top_k=args.top,
          k_to_50pct=k50, readout_cv_r2=cv_r2)
    for f in ("sae_attribution_features.csv", "sae_attribution_cumulative.csv",
              "sae_attribution_summary.csv", "sae_attribution_controls.csv"):
        print(f"-> {sae_dir / f}")


if __name__ == "__main__":
    main()
