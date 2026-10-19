#!/usr/bin/env python3
"""The nominated feature against the other cognitive targets (exploratory).

The nomination was made for fluid intelligence. This asks whether the nominated
autocorrelation signal also extends to the related cognitive outcomes, and whether it
does so beyond what those outcomes share with fluid intelligence.

Nothing is selected here. The features are the ones already nominated for fluid
intelligence; for every target, each model is fitted on the 656 training subjects and
evaluated once on the 280 held out, exactly as distill_confirm_hcp.py does for PMAT24.

Two readouts per target:

  raw            the Pearson correlation of ac1_CentExe itself with the target, on the
                 holdout. No fitting, so the sign is interpretable.
  partial        the same correlation controlling for PMAT24. Every cognitive score
                 shares variance with fluid intelligence, so a raw correlation with,
                 say, crystallized cognition can be carried entirely by that overlap.
                 The partial asks whether the feature adds anything beyond it.

Plus, for context, the holdout r of the frozen embedding and of the three nominated
features refitted to each target.

Usage:
    python src/04_interpretability/specificity_hcp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "04_interpretability"))
from common import load_config, resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402
from distill_confirm_hcp import (SIGNAL, fisher_ci, frozen_embedding_holdout,  # noqa: E402
                                 ridge)

FLUID = "PMAT24_A_CR"
TARGETS = [
    "PMAT24_A_CR", "CogFluidComp_Unadj", "ListSort_Unadj", "PicSeq_Unadj",
    "CogCrystalComp_Unadj", "PicVocab_Unadj", "ReadEng_Unadj",
    "ProcSpeed_Unadj", "CogTotalComp_Unadj",
]
FEATURE = "ac1_CentExe"
NOMINATED = ["ac1_CentExe", "fc_mean", "fc_sd"]
CEN_TIMESCALE = ["ac1_CentExe", "slope_CentExe", "falff_CentExe"]


def partial_r(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Correlation of x and y after regressing z out of both."""
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(stats.pearsonr(rx, ry)[0])


def partial_p(r: float, n: int, k: int = 1) -> float:
    df = n - 2 - k
    t = r * np.sqrt(df / max(1e-12, 1 - r * r))
    return float(2 * stats.t.sf(abs(t), df))


def main() -> None:
    dataset = "hcp_ya"
    cfg = load_config(dataset)
    interp = ROOT / "results" / dataset / "brainlm" / "interpretability"
    split_dir = ROOT / "results" / dataset / "splits"
    train_ids = [l.strip() for l in open(split_dir / "train_subjects.txt") if l.strip()]
    hold_ids = [l.strip() for l in open(split_dir / "holdout_subjects.txt") if l.strip()]

    feat = pd.read_csv(interp / "signal_features_per_window.csv.gz")
    subj = feat.pop("subject").astype(str)
    F = feat.groupby(subj.values).mean()
    F = F[[c for c in F.columns if c.startswith(SIGNAL)]]

    pheno = pd.read_csv(resolve(cfg["metadata"]["phenotype"]), low_memory=False)
    pheno["Subject"] = pheno["Subject"].astype(str)
    P = pheno.set_index("Subject")

    rows = []
    for target in TARGETS:
        def keep(ids):
            return [s for s in ids if s in F.index and pd.notna(P[target].get(s, np.nan))
                    and pd.notna(P[FLUID].get(s, np.nan))]
        tr, ho = keep(train_ids), keep(hold_ids)
        ytr_raw = P.loc[tr, target].astype(float).values
        yho_raw = P.loc[ho, target].astype(float).values
        lo, hi = ytr_raw.min(), ytr_raw.max()
        ytr, yho = (ytr_raw - lo) / (hi - lo), (yho_raw - lo) / (hi - lo)
        fluid_ho = P.loc[ho, FLUID].astype(float).values
        x_ho = F.loc[ho, FEATURE].values.astype(float)

        r_raw = float(stats.pearsonr(x_ho, yho)[0])
        _, _, p_raw = fisher_ci(r_raw, len(ho))
        if target == FLUID:
            r_part, p_part = np.nan, np.nan
        else:
            r_part = partial_r(x_ho, yho, fluid_ho)
            p_part = partial_p(r_part, len(ho))

        A = np.nan_to_num(F.loc[tr, NOMINATED].values.astype(float))
        B = np.nan_to_num(F.loc[ho, NOMINATED].values.astype(float))
        r_nom = float(stats.pearsonr(yho, ridge().fit(A, ytr).predict(B))[0])

        pm, _ = frozen_embedding_holdout(dataset, tr, ho, ytr, 5, 0)
        r_model = float(stats.pearsonr(yho, pm)[0])

        rows.append({
            "target": target, "n_train": len(tr), "n_holdout": len(ho),
            "r_with_fluid": float(stats.pearsonr(yho_raw, fluid_ho)[0]),
            "ac1_CentExe_r": r_raw, "ac1_CentExe_p": p_raw,
            "ac1_CentExe_partial_r": r_part, "ac1_CentExe_partial_p": p_part,
            "nominated_holdout_r": r_nom, "model_holdout_r": r_model,
        })
        print(f"{target:<22} r(fluid) {rows[-1]['r_with_fluid']:+.2f}  "
              f"ac1_CentExe r {r_raw:+.3f} (p {p_raw:.1e})  partial {r_part:+.3f}  "
              f"nominated {r_nom:+.3f}  model {r_model:+.3f}", flush=True)

    # The central executive timescale features against fluid intelligence, per split:
    # the correlations the Discussion reports, from the held-out subjects.
    tl = []
    for split, ids in (("train", train_ids), ("holdout", hold_ids)):
        ids = [s for s in ids if s in F.index and pd.notna(P[FLUID].get(s, np.nan))]
        y = P.loc[ids, FLUID].astype(float).values
        for c in CEN_TIMESCALE:
            r, p = stats.pearsonr(F.loc[ids, c].values.astype(float), y)
            tl.append({"split": split, "n": len(ids), "feature": c,
                       "r": float(r), "p": float(p)})
            print(f"{split:<8} n={len(ids)}  {c:<14} r {r:+.3f}  p {p:.1e}")

    df = pd.DataFrame(rows)
    out = ROOT / "results" / dataset / "brainlm" / "metrics"
    df.to_csv(out / "specificity_holdout.csv", index=False)
    pd.DataFrame(tl).to_csv(out / f"cen_timescale_{FLUID}.csv", index=False)
    stamp(out, "src/04_interpretability/specificity_hcp.py", ROOT,
          feature=FEATURE, nominated=NOMINATED, controlled_for=FLUID,
          cen_timescale=CEN_TIMESCALE)
    print(f"\n-> {out}/specificity_holdout.csv")
    print(f"-> {out}/cen_timescale_{FLUID}.csv")


if __name__ == "__main__":
    main()
