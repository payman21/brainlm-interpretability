#!/usr/bin/env python3
"""Plan section 4.1 -- predict a subject-level target from FROZEN BrainLM embeddings.

No fine-tuning. The encoder never moves; only the head is trained. This is the
first branch of plan section 1: if it predicts well, go straight to the
interpretability work in section 7; if not, fall back to fine-tuning (4.2).

Two aggregations, because a subject has ~24 windows and one prediction:

  subject_mean       average the window embeddings, then predict once
  window_then_mean   predict per window, then average the predictions

Two embeddings (``cls``, ``mean``) and two heads (ridge, MLP) are crossed with
those, so one run fills a single comparable table.

Folds are subject-level: every window of a subject lands in the same fold, so
no subject is partly trained on and partly held out. NOTE this does NOT control
family structure -- HCP-YA is a twin study and Family_ID is restricted, so a
held-out subject's twin may sit in training. See plan section 6.2.

The target is min-max scaled to [0, 1] following BrainLM section 3.4, which also
makes MSE directly comparable to their Table 2 (neuroticism: 0.086 for frozen
embeddings, 0.072 fine-tuned, both on UK Biobank).

Usage:
    python src/03_evaluation/predict_frozen_hcp.py
    python src/03_evaluation/predict_frozen_hcp.py --target NEOFAC_E --folds 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
from common import load_config, resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402

ALPHAS = np.logspace(-2, 6, 25)


def load_embeddings(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def build_heads(seed: int) -> dict:
    return {
        # Standardise first: embedding dimensions are on very different scales
        # and ridge's penalty is not scale-invariant.
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
        # The 3-layer MLP of BrainLM section 3.4, as a frozen-feature head.
        "mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(256, 128),
                alpha=1e-3,
                max_iter=2000,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=seed,
            ),
        ),
    }


def evaluate(X_sub, y, groups_idx, X_win, win_subject, folds, seed, mode, head_name, seed_head):
    """Out-of-fold predictions for one (aggregation, head) cell."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(np.arange(len(y))):
        head = build_heads(seed_head)[head_name]
        if mode == "subject_mean":
            head.fit(X_sub[tr], y[tr])
            oof[te] = head.predict(X_sub[te])
        else:
            # Train on individual windows of training subjects, then average the
            # per-window predictions within each held-out subject.
            tr_set, te_set = set(tr), set(te)
            m_tr = np.array([s in tr_set for s in win_subject])
            head.fit(X_win[m_tr], y[win_subject[m_tr]])
            m_te = np.array([s in te_set for s in win_subject])
            pw = head.predict(X_win[m_te])
            subj_of = win_subject[m_te]
            for s in te:
                oof[s] = pw[subj_of == s].mean()
    r2 = 1.0 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {
        "r2": float(r2),
        "r": float(stats.pearsonr(y, oof)[0]),
        "mse": float(((y - oof) ** 2).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--target", default=None, help="default: label_column from the config")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embeddings", type=Path,
                    default=ROOT / "results/hcp_ya/brainlm/embeddings/frozen_embeddings.npz")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    target = args.target or cfg["label_column"]

    z = load_embeddings(args.embeddings)
    subj_col = z["subject_id"].astype(str)
    print(f"embeddings: {len(subj_col)} windows, {len(set(subj_col))} subjects, "
          f"keys {sorted(z)}")

    # Analysis cohort: QC code C excluded (plan section 5.5).
    cohort = [l.strip() for l in open(resolve(cfg["cohort"]["subject_list"])) if l.strip()]
    pheno = pd.read_csv(resolve(cfg["metadata"]["phenotype"]), low_memory=False)
    pheno["Subject"] = pheno["Subject"].astype(str)
    tgt = pheno.set_index("Subject")[target]

    keep = [s for s in cohort if s in set(subj_col) and pd.notna(tgt.get(s, np.nan))]
    print(f"cohort {len(cohort)} -> {len(keep)} with embeddings and a non-null {target}")

    subj_index = {s: i for i, s in enumerate(keep)}
    win_mask = np.array([s in subj_index for s in subj_col])
    win_subject = np.array([subj_index[s] for s in subj_col[win_mask]])

    y_raw = np.array([tgt[s] for s in keep], dtype=float)
    lo, hi = y_raw.min(), y_raw.max()
    y = (y_raw - lo) / (hi - lo)          # min-max to [0,1], per BrainLM section 3.4
    print(f"target {target}: n={len(y)} raw range [{lo:g}, {hi:g}] "
          f"mean {y_raw.mean():.2f} sd {y_raw.std():.2f}")
    print(f"windows used: {win_mask.sum()} (mean {win_mask.sum()/len(keep):.1f} per subject)")

    rows = []
    for emb_name in ["cls", "mean"]:
        X_win = z[emb_name][win_mask].astype(np.float64)
        X_sub = np.stack([X_win[win_subject == i].mean(0) for i in range(len(keep))])
        for mode in ["subject_mean", "window_then_mean"]:
            for head_name in ["ridge", "mlp"]:
                m = evaluate(X_sub, y, None, X_win, win_subject,
                             args.folds, args.seed, mode, head_name, args.seed)
                rows.append({"embedding": emb_name, "aggregation": mode,
                             "head": head_name, **m})
                print(f"  {emb_name:>4} {mode:>16} {head_name:>5}: "
                      f"R2 {m['r2']:+.4f}  r {m['r']:+.4f}  MSE {m['mse']:.4f}", flush=True)

    df = pd.DataFrame(rows).sort_values("r2", ascending=False)
    print(f"\n=== {target}, frozen BrainLM, {len(keep)} subjects, "
          f"{args.folds}-fold subject-level CV ===")
    print(df.to_string(index=False))
    best = df.iloc[0]
    print(f"\nbest: {best['embedding']}/{best['aggregation']}/{best['head']}  "
          f"R2 {best['r2']:+.4f}  MSE {best['mse']:.4f}")
    # NOTE: MSE is NOT comparable to BrainLM Table 2. Min-max scaling fixes the
    # range, not the variance, and R2 = 1 - MSE/var(y). Their UKB neuroticism has
    # a different spread from HCP's NEOFAC_N, so only R2 travels between them.
    print(f"target variance after scaling: {y.var():.4f} "
          f"(R2 = 1 - MSE/var, so MSE alone does not compare across cohorts)")

    out_dir = ROOT / "results" / args.dataset / "brainlm" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"frozen_prediction_{target}.csv", index=False)
    stamp(out_dir, "src/03_evaluation/predict_frozen_hcp.py", ROOT,
          target=target, n_subjects=len(keep), n_windows=int(win_mask.sum()),
          folds=args.folds, best_r2=float(best.r2))
    print(f"\n-> {out_dir / f'frozen_prediction_{target}.csv'}")


if __name__ == "__main__":
    main()
