#!/usr/bin/env python3
"""Can the frozen embeddings predict anything at all?

Run alongside ``predict_frozen_hcp.py``. A near-zero R2 on the primary target is
only interpretable if the same embeddings demonstrably carry signal, so this
sweeps several targets plus two controls and reports them in one table.

  positive control   sex from resting-state fMRI is reliably decodable, so a
                     near-chance score here means the embeddings are broken
                     rather than that the target is absent
  acquisition control phase-encoding direction (LR vs RL) is pure scanner
                     configuration and holds no biology. A high score here would
                     mean the embeddings are dominated by acquisition, which
                     would undermine any later interpretability claim

Usage:
    python src/03_evaluation/frozen_controls_hcp.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
from common import load_config, resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402

TARGETS = [
    "NEOFAC_N", "NEOFAC_E", "NEOFAC_O", "NEOFAC_A", "NEOFAC_C",
    "CogTotalComp_Unadj", "CogFluidComp_Unadj", "CogCrystalComp_Unadj",
    "PMAT24_A_CR", "ListSort_Unadj", "ProcSpeed_Unadj", "ReadEng_Unadj",
    "PicVocab_Unadj", "PicSeq_Unadj",
    "PSQI_Score", "LifeSatisf_Unadj", "PercStress_Unadj",
    "FearAffect_Unadj", "Sadness_Unadj",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--embedding", default="cls", choices=["cls", "mean"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    z = np.load(ROOT / f"results/{args.dataset}/brainlm/embeddings/frozen_embeddings.npz",
                allow_pickle=True)
    sid, run = z["subject_id"].astype(str), z["run"].astype(str)

    cohort = [l.strip() for l in open(resolve(cfg["cohort"]["subject_list"])) if l.strip()]
    pheno = pd.read_csv(resolve(cfg["metadata"]["phenotype"]), low_memory=False)
    pheno["Subject"] = pheno["Subject"].astype(str)
    pheno = pheno.set_index("Subject")

    keep = [s for s in cohort if s in set(sid)]
    idx = {s: i for i, s in enumerate(keep)}
    m = np.array([s in idx for s in sid])
    si = np.array([idx[s] for s in sid[m]])
    X = z[args.embedding][m].astype(np.float64)
    X_sub = np.stack([X[si == i].mean(0) for i in range(len(keep))])
    kf = KFold(args.folds, shuffle=True, random_state=args.seed)
    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 6, 25)))

    print(f"{len(keep)} subjects | {m.sum()} windows | embedding '{args.embedding}' | "
          f"{args.folds}-fold subject-level CV\n")

    rows = []
    for t in TARGETS:
        if t not in pheno.columns:
            continue
        y = pheno.loc[keep, t].values.astype(float)
        ok = np.isfinite(y)
        p = cross_val_predict(ridge(), X_sub[ok], y[ok], cv=kf)
        r2 = 1.0 - ((y[ok] - p) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
        r = stats.pearsonr(y[ok], p)[0]
        rows.append({"target": t, "kind": "continuous", "n": int(ok.sum()),
                     "r2": float(r2), "r": float(r), "score": float(r2)})
        print(f"  {t:<22} n={ok.sum():>4}  R2 {r2:+.4f}  r {r:+.3f}")

    # --- controls -------------------------------------------------------
    g = (pheno.loc[keep, "Gender"].values == "F").astype(int)
    p = cross_val_predict(make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=3000, C=0.1)),
                          X_sub, g, cv=kf)
    acc_sex = balanced_accuracy_score(g, p)
    rows.append({"target": "Sex (F/M)", "kind": "positive_control", "n": len(g),
                 "r2": np.nan, "r": np.nan, "score": float(acc_sex)})

    pe = (np.char.find(run[m].astype(str), "_LR") >= 0).astype(int)
    groups = np.array([keep[i] for i in si])
    p = cross_val_predict(make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=1000, C=0.1)),
                          X, pe, cv=GroupKFold(args.folds), groups=groups)
    acc_pe = balanced_accuracy_score(pe, p)
    rows.append({"target": "PhaseEncode (LR/RL)", "kind": "acquisition_control",
                 "n": len(pe), "r2": np.nan, "r": np.nan, "score": float(acc_pe)})

    print(f"\n  {'Sex (F/M)':<22} n={len(g):>4}  balanced acc {acc_sex:.3f}  (chance 0.500)")
    print(f"  {'PhaseEncode (LR/RL)':<22} n={len(pe):>4}  balanced acc {acc_pe:.3f}  (chance 0.500)")

    df = pd.DataFrame(rows)
    out = ROOT / "results" / args.dataset / "brainlm" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"frozen_controls_{args.embedding}.csv", index=False)
    stamp(out, "src/03_evaluation/frozen_controls_hcp.py", ROOT,
          embedding=args.embedding, n_subjects=len(keep), folds=args.folds,
          sex_balanced_acc=float(acc_sex), phase_encode_balanced_acc=float(acc_pe))
    print(f"\n-> {out / f'frozen_controls_{args.embedding}.csv'}")


if __name__ == "__main__":
    main()
