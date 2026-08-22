#!/usr/bin/env python3
"""What did PRETRAINING contribute to the probe? Derived from probe_signal_features.csv.

A probe score read on its own is not about pretraining. The same probe applied to a
randomly initialised BrainLM already recovers global functional connectivity at
R2 0.760, because a random nonlinear mixture of the input does not destroy the
information a linear probe is looking for. Only the difference between the two
columns is attributable to what pretraining learned.

Two ways to express that difference, and they do not rank the families the same way:

    gain            r2_pretrained - r2_untrained
                    The raw difference. Compressed near the ceiling: a family whose
                    untrained baseline is already 0.83 cannot gain more than 0.17,
                    so this metric quietly penalises the families that start high.

    error_removed   (resid_untrained - resid_pretrained) / resid_untrained,
                    where resid = 1 - r2
                    The share of the untrained model's REMAINING error that
                    pretraining removes. Scale-free, so families with different
                    baselines are comparable.

Network FC has the larger `gain` (+0.197 against +0.149) because its untrained
baseline is lower (0.587 against 0.760); global FC has the larger `error_removed`
(62% against 48%). Neither ordering is robust and no claim should rest on which of
the two connectivity families is first. What IS robust is the separation between
blocks: connectivity 48-62%, timescale and spectral shape 12-24%, post-scaling
amplitude 3%.

`random_proj` is carried through as a third column because it is the floor. It is a
fixed linear projection of the window, so it recovers essentially nothing (<= 0.007
everywhere), which is what rules out the reading that these targets fall out of any
random 512-d compression.

Reads:  results/<dataset>/brainlm/interpretability/probe_signal_features.csv
Writes: probe_pretraining_gain.csv             one row per target
        probe_pretraining_gain_by_family.csv   one row per family

Usage:
    python src/04_interpretability/probe_pretraining_gain.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common_provenance import stamp  # noqa: E402

# Presentation order: the families the model can see, then the ones preprocessing
# removed from its input, then provenance of the window itself.
ORDER = ["fc", "netfc", "ac1", "slope", "falff", "var",
         "rawalff", "rawvar", "iqr",
         "window", "runphase", "session"]

LABEL = {
    "fc": "Global FC", "netfc": "Network FC", "ac1": "Lag-1 autocorrelation",
    "slope": "Aperiodic spectral slope", "falff": "fALFF",
    "var": "Amplitude, after scaling", "rawalff": "Raw ALFF",
    "rawvar": "Raw parcel SD", "iqr": "Parcel IQR (the divisor)",
    "window": "Window index in run", "runphase": "Phase-encoding direction",
    "session": "Session (REST1 vs REST2)",
}


def error_removed(r2_hi: np.ndarray, r2_lo: np.ndarray) -> np.ndarray:
    """Share of the baseline's unexplained variance that the better model removes.

    Undefined when the baseline leaves no error to remove, which cannot happen at
    these values but is guarded rather than left to produce a silent inf.
    """
    resid_lo, resid_hi = 1.0 - r2_lo, 1.0 - r2_hi
    return np.where(resid_lo > 0, (resid_lo - resid_hi) / resid_lo, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--pretrained", default="frozen_mean")
    ap.add_argument("--baseline", default="random_init_mean")
    args = ap.parse_args()

    d = ROOT / "results" / args.dataset / "brainlm" / "interpretability"
    src = d / "probe_signal_features.csv"
    if not src.exists():
        raise SystemExit(f"{src} missing. Run probe_signal_features_hcp.py first.")
    long = pd.read_csv(src)

    have = set(long["source"])
    for s in (args.pretrained, args.baseline):
        if s not in have:
            raise SystemExit(
                f"source {s!r} is not in {src.name}; it holds {sorted(have)}. "
                "The untrained baseline needs random_init_embeddings.npz -- see "
                "extract_embeddings.py --random-init.")

    wide = long.pivot_table(index=["feature", "family"], columns="source",
                            values="r2").reset_index()
    wide = wide.rename(columns={args.pretrained: "r2_pretrained",
                                args.baseline: "r2_untrained"})
    wide["gain"] = wide.r2_pretrained - wide.r2_untrained
    wide["error_removed"] = error_removed(wide.r2_pretrained.values,
                                          wide.r2_untrained.values)

    keep = ["feature", "family", "r2_pretrained", "r2_untrained", "gain", "error_removed"]
    if "random_proj" in have:
        keep.append("random_proj")
        wide = wide.rename(columns={"random_proj": "r2_random_proj"})
        keep[-1] = "r2_random_proj"
    if "frozen_cls" in have:
        wide = wide.rename(columns={"frozen_cls": "r2_pretrained_cls"})
        keep.append("r2_pretrained_cls")
    wide = wide[keep].sort_values("error_removed", ascending=False)

    # Family level. `error_removed` is computed from the family MEAN r2 of each
    # source -- the same arithmetic as the summary table -- and the mean of the
    # per-target values is carried alongside it, because the two differ and a reader
    # should be able to see by how much rather than have to trust one of them.
    agg = {c: "mean" for c in wide.columns if c.startswith("r2_")}
    fam = wide.groupby("family").agg({**agg, "feature": "count"})
    fam = fam.rename(columns={"feature": "n_targets"})
    fam["gain"] = fam.r2_pretrained - fam.r2_untrained
    fam["error_removed"] = error_removed(fam.r2_pretrained.values, fam.r2_untrained.values)
    fam["error_removed_mean_of_targets"] = wide.groupby("family").error_removed.mean()
    fam = fam.reindex([f for f in ORDER if f in fam.index])
    fam.insert(0, "label", [LABEL.get(f, f) for f in fam.index])

    show = fam[["label", "n_targets", "r2_pretrained", "r2_untrained", "gain",
                "error_removed"]].copy()
    show["error_removed"] = (show.error_removed * 100).round(1)
    print(f"=== what pretraining contributed, {args.pretrained} vs {args.baseline} ===")
    print(show.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print("\nerror_removed is the share of the untrained model's remaining error "
          "that pretraining removes, in %.")

    wide.to_csv(d / "probe_pretraining_gain.csv", index=False)
    fam.to_csv(d / "probe_pretraining_gain_by_family.csv")
    stamp(d, "src/04_interpretability/probe_pretraining_gain.py", ROOT,
          pretrained=args.pretrained, baseline=args.baseline,
          n_targets=int(len(wide)), n_families=int(len(fam)))
    print(f"\n-> {d}/probe_pretraining_gain.csv")
    print(f"-> {d}/probe_pretraining_gain_by_family.csv")


if __name__ == "__main__":
    main()
