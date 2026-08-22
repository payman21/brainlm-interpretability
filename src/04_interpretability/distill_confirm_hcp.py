#!/usr/bin/env python3
"""Steps 8-9 -- distillation as CONFIRMATION, then one look at the holdout.

What changed, and why this is a different script from distill_hcp.py
-------------------------------------------------------------------
`distill_hcp.py` searched 51 candidate features on all 936 subjects, found lag-1
autocorrelation wins, and was also the only evidence that it wins. The winner was
chosen after seeing the whole table, the margin over the runner-up was 0.002 R2, and
everything was cross-validated inside one cohort. That is a hypothesis generated and
tested on the same data.

Three things fix it, and all three are in this script:

  NOMINATION   The hypothesis now comes from somewhere else. The SAE attribution
               (sae_attribution_hcp.py, step 5) selected 29 features accounting for
               58.7% of the model's prediction variance; characterising them
               (step 7) found 27 of 29 concentrated in the central executive network,
               with lag-1 autocorrelation enriched 2.7x and global FC 3.0x. That is a
               feature set fixed BEFORE this script runs, by a method that never saw a
               hand-crafted feature. `nominated_sae` below is that set.

  BOOTSTRAPS   Following the reference study: 100 bootstraps of the training split,
               retain features selected in at least 90%, and report the selection
               counts next to the performance table. With 51 candidates and 0.002 R2
               between first and second place, a single fit does not establish an
               ordering.

  HOLDOUT      One evaluation, at the end, on the 280 subjects set aside before the
               SAE was even retrained. No iterating on it. Every feature set declared
               in SETS is evaluated once and all of them are reported, so nothing can
               be quietly dropped for reading badly.

Read the holdout column as the result and the CV column as context, not the reverse.

CAVEAT THAT TRAVELS WITH THE HOLDOUT NUMBER
-------------------------------------------
The split is subject-level but NOT family-level: HCP-YA is a twin study and Family_ID
is restricted, so a held-out subject's twin may sit in training. See
results/hcp_ya/splits/split_manifest.json.

Usage:
    python src/04_interpretability/distill_confirm_hcp.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
from common import load_config, resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402

ALPHAS = np.logspace(-2, 6, 25)

# The five scale-invariant signal families. The feature file also carries the probe's
# deliberately-unlearnable columns -- absolute amplitude (var_, rawvar_, rawalff_,
# iqr_) and provenance (window_index, runphase_, session_) -- which exist to be probed
# and fail. Distilling onto those would be distilling onto preprocessing artefacts.
SIGNAL = ("fc_", "netfc_", "ac1_", "slope_", "falff_")

# Every set is declared HERE, before the script runs, and every one is reported.
SETS = {
    # The nomination: what the SAE attribution pointed at, and the only set in this
    # table that was fixed by an independent method rather than by inspecting a
    # distillation ranking. ac1 in the central executive network plus global FC.
    "nominated_sae":   lambda c: c in ("ac1_CentExe", "fc_mean", "fc_sd"),
    # The same nomination widened to the whole central executive column, to show the
    # claim does not hinge on one parcel grouping.
    "centexe_only":    lambda c: c.endswith("CentExe") or c in ("fc_mean", "fc_sd"),
    # The families, for continuity with distill_hcp.py.
    "fc_global":       lambda c: c.startswith("fc_"),
    "netfc":           lambda c: c.startswith("netfc_"),
    "ac1":             lambda c: c.startswith("ac1_"),
    "spectral":        lambda c: c.startswith("slope_") or c.startswith("falff_"),
    "all_explicit":    lambda c: c.startswith(SIGNAL),
}


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def boot_ci(y: np.ndarray, p: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for holdout R2, resampling SUBJECTS.

    Required, not optional: at 280 subjects and R2 around 0.05 the interval is wide
    enough that differences between feature sets in the third decimal are not
    differences at all. The reference study puts a 95% CI on every headline number and
    this table is unreadable without one.
    """
    rng = np.random.default_rng(seed)
    vals = [r2(y[i], p[i]) for i in
            (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fisher_ci(r: float, n: int) -> tuple[float, float, float]:
    """95% CI and two-sided p for a correlation, via Fisher's z.

    Both r and R2 are reported for every row because they disagree here, and the
    disagreement is itself informative rather than a reason to pick one. At n=280 the
    bootstrap R2 interval spans zero for every feature set while the correlation
    interval excludes it: R2 is penalised by any calibration error in the predictions,
    the correlation is scale-free. `r` is also the metric BrainLM's own paper and the
    HCP behavioural-prediction literature report, so it is the primary column by
    convention and not by selection. Report both.
    """
    z, se = np.arctanh(r), 1.0 / np.sqrt(max(n - 3, 1))
    return (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)),
            float(2 * stats.norm.sf(abs(z) / se)))


def frozen_embedding_holdout(dataset: str, tr_ids: list[str], ho_ids: list[str],
                             ytr: np.ndarray, folds: int, seed: int):
    """THE ANCHOR: what BrainLM itself scores on the same holdout, same protocol.

    Every "% of the model" figure in this project came from in-cohort cross-validation
    on all 936 subjects. Without the model's own number on these 280 subjects there is
    nothing for the distilled feature sets to be a percentage OF, so this row is what
    makes the rest of the table interpretable.

    Uses `mean` / `window_then_mean` / ridge -- the readout that produced the published
    R2 0.0748 -- fitted on the 656 and evaluated once on the 280. Reads the per-window
    token-mean embeddings from sae_window_means.npz, which were computed in the same
    pass as everything else and are bit-identical to frozen_embeddings.npz.
    """
    wm = np.load(ROOT / f"results/{dataset}/brainlm/interpretability/sae/sae_window_means.npz",
                 allow_pickle=True)
    E = np.asarray(wm["emb_mean"]).astype(np.float64)
    subj = np.asarray(wm["subject"]).astype(str)
    tr, ho = set(tr_ids), set(ho_ids)
    m_tr = np.array([s in tr for s in subj])
    m_ho = np.array([s in ho for s in subj])
    ytr_of = {s: v for s, v in zip(tr_ids, ytr)}
    y_win = np.array([ytr_of[s] for s in subj[m_tr]])

    model = ridge().fit(E[m_tr], y_win)
    # window_then_mean: predict per window, average within subject. Identical to
    # predicting on the subject-mean embedding for a linear readout, verified at 3e-16.
    pw = model.predict(E[m_ho])
    sh = subj[m_ho]
    pred = np.array([pw[sh == s].mean() for s in ho_ids])

    # Training-split CV for context, folds grouped by subject.
    rng = np.random.default_rng(seed)
    fold_of = {s: f for s, f in zip(tr_ids, rng.permutation(len(tr_ids)) % folds)}
    oof = np.full(len(tr_ids), np.nan)
    fw = np.array([fold_of[s] for s in subj[m_tr]])
    for f in range(folds):
        mdl = ridge().fit(E[m_tr][fw != f], y_win[fw != f])
        q = mdl.predict(E[m_tr][fw == f])
        s_te = subj[m_tr][fw == f]
        for j, s in enumerate(tr_ids):
            if fold_of[s] == f:
                oof[j] = q[s_te == s].mean()
    return pred, oof


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--target", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstraps", type=int, default=100)
    ap.add_argument("--selection-threshold", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=2000,
                    help="bootstrap resamples for the holdout confidence intervals")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    target = args.target or cfg["label_column"]
    interp = ROOT / "results" / args.dataset / "brainlm" / "interpretability"
    split_dir = ROOT / "results" / args.dataset / "splits"

    train_ids = [l.strip() for l in open(split_dir / "train_subjects.txt") if l.strip()]
    hold_ids = [l.strip() for l in open(split_dir / "holdout_subjects.txt") if l.strip()]

    feat = pd.read_csv(interp / "signal_features_per_window.csv.gz")
    subj = feat.pop("subject").astype(str)      # int64 through CSV; the index is str
    F = feat.groupby(subj.values).mean()
    F = F[[c for c in F.columns if c.startswith(SIGNAL)]]
    print(f"explicit features: {F.shape[0]} subjects x {F.shape[1]} signal features")

    pheno = pd.read_csv(resolve(cfg["metadata"]["phenotype"]), low_memory=False)
    pheno["Subject"] = pheno["Subject"].astype(str)
    tgt = pheno.set_index("Subject")[target]

    def prep(ids):
        keep = [s for s in ids if s in F.index and pd.notna(tgt.get(s, np.nan))]
        return keep, F.loc[keep], np.array([float(tgt[s]) for s in keep])

    tr_ids, Xtr_df, ytr_raw = prep(train_ids)
    ho_ids, Xho_df, yho_raw = prep(hold_ids)
    # Min-max from the TRAINING targets only. R2 is invariant to an affine transform of
    # y, so this changes nothing except that the holdout's range never leaks in.
    lo, hi = ytr_raw.min(), ytr_raw.max()
    ytr = (ytr_raw - lo) / (hi - lo)
    yho = (yho_raw - lo) / (hi - lo)
    print(f"train {len(tr_ids)} subjects | holdout {len(ho_ids)} subjects | target {target}")

    # --- step 8: bootstrap feature selection, on the training split only -------------
    pool = list(Xtr_df.columns)
    Xp = np.nan_to_num(Xtr_df[pool].values.astype(float))
    rng = np.random.default_rng(args.seed)
    counts = np.zeros(len(pool), int)
    for b in range(args.bootstraps):
        idx = rng.integers(0, len(Xp), len(Xp))
        if len(np.unique(ytr[idx])) < 3:
            continue
        m = make_pipeline(StandardScaler(),
                          LassoCV(n_alphas=40, cv=5, max_iter=20000,
                                  random_state=args.seed)).fit(Xp[idx], ytr[idx])
        counts += (m[-1].coef_ != 0).astype(int)
        if (b + 1) % 25 == 0:
            print(f"  bootstrap {b + 1}/{args.bootstraps}", flush=True)
    sel = pd.DataFrame({"feature": pool, "selection_count": counts,
                        "selection_frac": counts / args.bootstraps})
    sel = sel.sort_values("selection_count", ascending=False)
    retained = sel.loc[sel.selection_frac >= args.selection_threshold, "feature"].tolist()
    print(f"\nbootstrap selection: {len(retained)}/{len(pool)} features retained at "
          f">={args.selection_threshold:.0%} of {args.bootstraps} bootstraps")
    print(sel.head(15).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # The selected set joins the declared ones. It is a legitimate row because the
    # selection saw only the training split.
    sets = dict(SETS)
    if retained:
        sets[f"bootstrap_{int(args.selection_threshold*100)}pct"] = lambda c, r=set(retained): c in r

    # --- step 9: cross-validate on train, then ONE evaluation on the holdout --------
    kf = KFold(args.folds, shuffle=True, random_state=args.seed)
    rows = []

    # The anchor first, so every row below can be read as a fraction of it.
    ho_pred_m, cv_pred_m = frozen_embedding_holdout(
        args.dataset, tr_ids, ho_ids, ytr, args.folds, args.seed)
    lo_m, hi_m = boot_ci(yho, ho_pred_m, args.boot, args.seed)
    rm = float(stats.pearsonr(yho, ho_pred_m)[0])
    rlo, rhi, rp = fisher_ci(rm, len(yho))
    rows.append({"feature_set": "BrainLM frozen embedding (the model)", "n_features": 512,
                 "train_cv_r2": r2(ytr, cv_pred_m),
                 "train_cv_r": float(stats.pearsonr(ytr, cv_pred_m)[0]),
                 "holdout_r2": r2(yho, ho_pred_m), "holdout_r": rm,
                 "holdout_r2_lo": lo_m, "holdout_r2_hi": hi_m,
                 "holdout_r_lo": rlo, "holdout_r_hi": rhi, "holdout_r_p": rp,
                 "pct_of_model": 100.0})
    base = rows[0]["holdout_r2"]
    print(f"\nANCHOR -- BrainLM itself on the same 280: holdout R2 {base:+.4f} "
          f"[{lo_m:+.4f}, {hi_m:+.4f}]  (train CV {rows[0]['train_cv_r2']:+.4f})\n")

    for name, fn in sets.items():
        cols = [c for c in pool if fn(c)]
        if not cols:
            continue
        A = np.nan_to_num(Xtr_df[cols].values.astype(float))
        B = np.nan_to_num(Xho_df[cols].values.astype(float))
        cv_pred = cross_val_predict(ridge(), A, ytr, cv=kf)
        model = ridge().fit(A, ytr)                 # fitted on ALL of the training split
        ho_pred = model.predict(B)                  # the one evaluation
        lo, hi = boot_ci(yho, ho_pred, args.boot, args.seed)
        rr = float(stats.pearsonr(yho, ho_pred)[0])
        rlo, rhi, rp = fisher_ci(rr, len(yho))
        rows.append({
            "feature_set": name, "n_features": len(cols),
            "train_cv_r2": r2(ytr, cv_pred), "train_cv_r": float(stats.pearsonr(ytr, cv_pred)[0]),
            "holdout_r2": r2(yho, ho_pred), "holdout_r": rr,
            "holdout_r2_lo": lo, "holdout_r2_hi": hi,
            "holdout_r_lo": rlo, "holdout_r_hi": rhi, "holdout_r_p": rp,
            "pct_of_model": 100.0 * r2(yho, ho_pred) / base if base > 0 else np.nan,
        })
        print(f"  {name:<22} n={len(cols):>3}  train CV R2 {rows[-1]['train_cv_r2']:+.4f}"
              f"  |  HOLDOUT R2 {rows[-1]['holdout_r2']:+.4f} "
              f"= {rows[-1]['pct_of_model']:.0f}% of model  |  r {rr:+.3f} "
              f"[{rlo:+.3f}, {rhi:+.3f}] p {rp:.1e}",
              flush=True)

    df = pd.DataFrame(rows)
    print("\n=== steps 8-9: confirmation, and one look at the holdout ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nThe holdout columns are ONE evaluation each, on 280 subjects that played no "
          "part in the split, the SAE, the attribution or the selection. Do not iterate.")

    out = ROOT / "results" / args.dataset / "brainlm" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"distill_confirm_{target}.csv", index=False)
    sel.to_csv(out / f"distill_bootstrap_selection_{target}.csv", index=False)
    stamp(out, "src/04_interpretability/distill_confirm_hcp.py", ROOT,
          target=target, n_train=len(tr_ids), n_holdout=len(ho_ids),
          bootstraps=args.bootstraps, selection_threshold=args.selection_threshold,
          n_retained=len(retained), retained=retained)
    print(f"\n-> {out}/distill_confirm_{target}.csv")
    print(f"-> {out}/distill_bootstrap_selection_{target}.csv")


if __name__ == "__main__":
    main()
