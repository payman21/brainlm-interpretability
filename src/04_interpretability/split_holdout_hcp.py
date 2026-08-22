#!/usr/bin/env python3
"""Step 0 -- carve a held-out subject set, BEFORE any interpretability step runs.

Why this has to come first
--------------------------
The distillation result in the current draft searched 51 candidate features, found
that lag-1 autocorrelation wins, and is also the only evidence that it wins. The
winner was chosen after seeing the whole table and the margin over the runner-up was
0.002 R2. That is a hypothesis generated and tested on the same data.

The fix is a set of subjects that plays no part in generating the hypothesis. It has
to be carved before the sparse autoencoder is retrained, because an SAE fitted on all
938 subjects has already seen the held-out ones -- unsupervised or not, its dictionary
is shaped by them, and every attribution built on that dictionary inherits the
contact. So: split first, then retrain the SAE on the training split alone, then
attribute, then distil, and touch the holdout exactly once at the end.

What this writes
----------------
    results/<dataset>/splits/train_subjects.txt      ~70%
    results/<dataset>/splits/holdout_subjects.txt    ~30%
    results/<dataset>/splits/split_manifest.json     counts, balance checks, seed

Both files are plain subject-id lists, one per line, and every downstream script takes
them by path rather than re-deriving a split of its own.

Stratification
--------------
By quintile of the target, so the two halves cannot differ in target distribution by
accident at n=938. The split uses the target's VALUE and nothing else -- no embedding,
no model output -- so it cannot favour any model. A balance check on the target and on
sex is printed and stored.

CAVEAT THAT MUST TRAVEL WITH THE HELDOUT NUMBER
-----------------------------------------------
HCP-YA is a twin study and `Family_ID` is restricted, so this split is subject-level
but NOT family-level: a held-out subject's twin may sit in the training split. The
holdout is therefore a genuine test of "unseen subject" and only a partial test of
"unseen person". Say so wherever the holdout number appears; it is the same limitation
the cross-validated numbers already carry.

Usage:
    python src/04_interpretability/split_holdout_hcp.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
from common import load_config, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--target", default=None, help="default: label_column from the config")
    ap.add_argument("--holdout-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-strata", type=int, default=5)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing split. Do not use casually: every result "
                         "downstream is conditioned on the split it was computed under")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    target = args.target or cfg["label_column"]
    out = ROOT / "results" / args.dataset / "splits"

    if (out / "train_subjects.txt").exists() and not args.force:
        raise SystemExit(
            f"{out}/train_subjects.txt already exists. A split is a commitment -- "
            "re-drawing it invalidates every attribution and distillation result "
            "computed under the old one. Pass --force only if you mean that.")

    # Exactly the cohort predict_frozen_hcp.py uses: QC-C excluded, has embeddings,
    # non-null target. Anything else and the split would not partition the analysis set.
    emb = np.load(ROOT / f"results/{args.dataset}/brainlm/embeddings/frozen_embeddings.npz",
                  allow_pickle=True)
    have_emb = set(emb["subject_id"].astype(str))
    cohort = [l.strip() for l in open(resolve(cfg["cohort"]["subject_list"])) if l.strip()]
    pheno = pd.read_csv(resolve(cfg["metadata"]["phenotype"]), low_memory=False)
    pheno["Subject"] = pheno["Subject"].astype(str)
    tgt = pheno.set_index("Subject")[target]

    subs = [s for s in cohort if s in have_emb and pd.notna(tgt.get(s, np.nan))]
    y = np.array([float(tgt[s]) for s in subs])
    print(f"cohort {len(cohort)} -> {len(subs)} with embeddings and a non-null {target}")

    # Stratify on target quintile. `duplicates="drop"` because a discrete score can
    # put an edge on a repeated value, which would otherwise raise.
    strata = pd.qcut(y, args.n_strata, labels=False, duplicates="drop")
    rng = np.random.default_rng(args.seed)
    hold = np.zeros(len(subs), bool)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        n_hold = int(round(args.holdout_frac * len(idx)))
        hold[idx[:n_hold]] = True

    train_ids = [subs[i] for i in np.where(~hold)[0]]
    hold_ids = [subs[i] for i in np.where(hold)[0]]

    # Balance checks. These are reported, not optimised -- a split tuned until the
    # t-test looked good would no longer be random.
    t, p = stats.ttest_ind(y[~hold], y[hold])
    checks = {
        "target_train_mean": float(y[~hold].mean()), "target_train_sd": float(y[~hold].std()),
        "target_holdout_mean": float(y[hold].mean()), "target_holdout_sd": float(y[hold].std()),
        "target_ttest_t": float(t), "target_ttest_p": float(p),
    }
    sex = pheno.set_index("Subject").get("Gender")
    if sex is not None:
        checks["frac_female_train"] = float(np.mean([sex.get(s) == "F" for s in train_ids]))
        checks["frac_female_holdout"] = float(np.mean([sex.get(s) == "F" for s in hold_ids]))

    print(f"\ntrain {len(train_ids)}  holdout {len(hold_ids)}  "
          f"({len(hold_ids)/len(subs):.1%} held out)")
    print(f"{target}: train {checks['target_train_mean']:.2f} "
          f"(sd {checks['target_train_sd']:.2f})  vs  holdout "
          f"{checks['target_holdout_mean']:.2f} (sd {checks['target_holdout_sd']:.2f})  "
          f"t={t:+.2f} p={p:.3f}")
    if "frac_female_train" in checks:
        print(f"fraction female: train {checks['frac_female_train']:.3f}  "
              f"holdout {checks['frac_female_holdout']:.3f}")
    assert not (set(train_ids) & set(hold_ids)), "splits overlap"
    assert len(train_ids) + len(hold_ids) == len(subs), "splits do not partition the cohort"

    out.mkdir(parents=True, exist_ok=True)
    (out / "train_subjects.txt").write_text("\n".join(train_ids) + "\n")
    (out / "holdout_subjects.txt").write_text("\n".join(hold_ids) + "\n")
    manifest = {
        "dataset": args.dataset, "target": target, "seed": args.seed,
        "holdout_frac_requested": args.holdout_frac,
        "holdout_frac_actual": len(hold_ids) / len(subs),
        "n_total": len(subs), "n_train": len(train_ids), "n_holdout": len(hold_ids),
        "n_strata": int(len(np.unique(strata))), "balance": checks,
        "caveat": "subject-level, NOT family-level: HCP-YA is a twin study and "
                  "Family_ID is restricted, so a held-out subject's twin may be in train.",
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n-> {out}/train_subjects.txt")
    print(f"-> {out}/holdout_subjects.txt")
    print(f"-> {out}/split_manifest.json")


if __name__ == "__main__":
    main()
