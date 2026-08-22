#!/usr/bin/env python3
"""Plan section 7.0b -- what does a TOKEN embedding know about where it came from?

The reference study [Goodfire/Prima Mente, 2026] probes single-fragment embeddings for
genomic **locus** and **region identity** and finds those recovered comparatively
poorly, against methylation and fragment length recovered well. That contrast is what
makes their probing table readable: it has rows that come out low.

The window-mean embedding cannot be asked this question -- it has already averaged
over every parcel and every temporal patch. So this probe works one level down, on the
last-layer activation of a single **token**. BrainLM tokenises a 424-parcel x
200-timepoint window into 4,240 tokens, one per (parcel, 20-timepoint patch) pair, so
each token has an unambiguous spatial and temporal address:

    parcel        which of the 424 A424 regions        (424-way)
    network       which of the 7 AA-7 networks         (7-way)
    xyz           the parcel's MNI centroid, in mm     (3 continuous)
    patch         which of the 10 temporal patches     (10-way, ordinal)

READ THE SCORES AGAINST THE BASELINE, NOT AGAINST CHANCE
--------------------------------------------------------
Parcel identity is not a discovery if it is recovered: BrainLM is HANDED each parcel's
xyz centroid as an additive input embedding, and each token is handed its patch index
the same way. Both are therefore recoverable by construction, and the only question
worth asking is whether pretraining changed anything about how they are held. That is
what `--random-init` is for: the identical probe on a randomly initialised encoder,
which has the same coordinate and position embeddings but learned nothing. The
difference between the two columns is the only part that is about pretraining.

Metrics, so that everything is comparable to the window-level probe:

    R2       out-of-fold, for xyz (mean over x/y/z) and for the ordinal patch index
    top-1    accuracy of a linear readout, against the 1/n_class chance rate

Both come from one ridge fit per fold: continuous targets directly, categorical ones by
regressing one-hot columns and taking the argmax. Folds are grouped by SUBJECT, so no
subject's tokens straddle a split.

Usage:
    python src/04_interpretability/probe_identity_tokens_hcp.py
    python src/04_interpretability/probe_identity_tokens_hcp.py --random-init
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
from common import resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402

ALPHAS = np.logspace(-2, 6, 25)


def fit_fold(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    mx, sx = Xtr.mean(0), Xtr.std(0)
    sx[sx == 0] = 1.0
    my, sy = Ytr.mean(0), Ytr.std(0)
    sy[sy == 0] = 1.0
    m = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
    m.fit((Xtr - mx) / sx, (Ytr - my) / sy)
    return m.predict((Xte - mx) / sx) * sy + my


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--random-init", action="store_true",
                    help="probe the randomly initialised encoder's tokens instead")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=200_000,
                    help="subsample for tractability; 424 one-hot columns over 500k rows "
                         "is a 500k x 424 dense target matrix")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = ROOT / "results" / args.dataset / "brainlm" / "interpretability"
    name = "token_activations_random_init.npz" if args.random_init else "token_activations.npz"
    p = d / name
    if not p.exists():
        raise SystemExit(
            f"{p} missing. Produce it with:\n"
            f"  python src/04_interpretability/extract_token_acts_hcp.py --dataset {args.dataset} "
            f"--device mps" + (" --random-init" if args.random_init else ""))

    z = np.load(p, allow_pickle=True)
    A = np.asarray(z["acts"]).astype(np.float64)
    meta = np.asarray(z["meta"])
    cols = list(z["meta_cols"])
    win_row = meta[:, cols.index("window_row")]
    parcel = meta[:, cols.index("parcel")]
    net7 = meta[:, cols.index("net7")]
    patch = meta[:, cols.index("patch")]
    subject = np.asarray(z["subject"]).astype(str)[win_row]

    rng = np.random.default_rng(args.seed)
    if len(A) > args.max_tokens:
        keep = rng.choice(len(A), args.max_tokens, replace=False)
        A, parcel, net7, patch, subject = A[keep], parcel[keep], net7[keep], patch[keep], subject[keep]

    xyz = np.loadtxt(resolve("external/atlases/A424/A424_Coordinates.dat"))[:, 1:]
    tag = "random_init" if args.random_init else "frozen"
    print(f"{tag}: {A.shape[0]:,} tokens x {A.shape[1]}-d, "
          f"{len(np.unique(subject))} subjects, {len(np.unique(parcel))} parcels")

    # One target matrix: 3 continuous xyz, the ordinal patch index, then one-hot
    # blocks for the three categorical readouts. Ridge is fitted once per fold over
    # all of it, and each block is scored with the metric that suits it.
    def onehot(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lev = np.unique(v)
        return (v[:, None] == lev[None, :]).astype(np.float64), lev

    Hpar, lev_par = onehot(parcel)
    Hnet, lev_net = onehot(net7)
    Hpat, lev_pat = onehot(patch)
    cont = np.column_stack([xyz[parcel], patch.astype(float)])
    Y = np.hstack([cont, Hpar, Hnet, Hpat])
    b = [0, 4, 4 + Hpar.shape[1], 4 + Hpar.shape[1] + Hnet.shape[1], Y.shape[1]]
    print(f"targets: {Y.shape[1]} columns "
          f"(4 continuous, {Hpar.shape[1]} parcel, {Hnet.shape[1]} network, {Hpat.shape[1]} patch)")

    P = np.empty_like(Y)
    for k, (tr, te) in enumerate(GroupKFold(args.folds).split(A, groups=subject), 1):
        P[te] = fit_fold(A[tr], Y[tr], A[te])
        print(f"  fold {k}/{args.folds}", flush=True)

    def r2(j0: int, j1: int) -> float:
        y, q = Y[:, j0:j1], P[:, j0:j1]
        return float(np.mean(1 - ((y - q) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)))

    def acc(j0: int, j1: int, truth: np.ndarray, lev: np.ndarray) -> float:
        return float((lev[P[:, j0:j1].argmax(1)] == truth).mean())

    res = [
        {"target": "parcel_xyz_mm", "kind": "continuous (3)", "metric": "R2",
         "value": r2(0, 3), "chance": 0.0},
        {"target": "patch_index", "kind": "ordinal (10)", "metric": "R2",
         "value": r2(3, 4), "chance": 0.0},
        {"target": "parcel_identity", "kind": "categorical (424)", "metric": "top-1",
         "value": acc(b[1], b[2], parcel, lev_par), "chance": 1.0 / len(lev_par)},
        {"target": "network_identity", "kind": "categorical (7)", "metric": "top-1",
         "value": acc(b[2], b[3], net7, lev_net), "chance": 1.0 / len(lev_net)},
        {"target": "patch_position", "kind": "categorical (10)", "metric": "top-1",
         "value": acc(b[3], b[4], patch, lev_pat), "chance": 1.0 / len(lev_pat)},
    ]
    df = pd.DataFrame(res).assign(source=tag, n_tokens=len(A))
    print(f"\n=== token-level identity probe, {tag} ===")
    print(df[["target", "kind", "metric", "value", "chance"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = d / f"probe_identity_tokens_{tag}.csv"
    df.to_csv(out, index=False)
    stamp(d, "src/04_interpretability/probe_identity_tokens_hcp.py", ROOT,
          weights=tag, n_tokens=int(len(A)), folds=args.folds,
          parcel_top1=float(df.loc[df.target == "parcel_identity", "value"].iloc[0]),
          patch_top1=float(df.loc[df.target == "patch_position", "value"].iloc[0]))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
