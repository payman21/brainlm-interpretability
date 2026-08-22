#!/usr/bin/env python3
"""Extract subsampled last-layer TOKEN activations, for SAE training (plan 7.4).

The full tensor is 22,482 windows x 4,241 tokens x 512 dims ~ 195 GB, so tokens are
subsampled. Standard practice for SAE training, and 500K activations is ample for a
512 -> 4096 dictionary.

For every sampled token we keep its provenance -- parcel, network, temporal patch,
subject, window -- because step 2 of the SAE analysis is characterising what each
learned feature responds to, and that needs to distinguish "fires on high
autocorrelation" from "fires on parcel 137" or "fires on the last patch". Without
provenance the dictionary is uninterpretable.

Tokens are parcel-major: token i is parcel i // n_patch, patch i % n_patch.

Usage:
    python src/04_interpretability/extract_token_acts_hcp.py --n-windows 5000 --tokens-per-window 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
from common import load_config, resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402

RUNS = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--model", default="brainlm")
    ap.add_argument("--n-windows", type=int, default=5000)
    ap.add_argument("--tokens-per-window", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--random-init",
        action="store_true",
        help="randomly initialised weights instead of the checkpoint. The baseline any "
        "token-level probe score must be read against: an untrained transformer over "
        "structured input is already a strong feature extractor.",
    )
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--subjects-file", type=Path, default=None,
                    help="draw the window sample only from these subject ids, one per "
                         "line. Pass results/<dataset>/splits/train_subjects.txt so the "
                         "dictionary's training sample is the full requested size rather "
                         "than what survives a filter applied afterwards")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    with (ROOT / "configs" / "models" / f"{args.model}.yaml").open() as fh:
        mcfg = yaml.safe_load(fh)
    sys.path.insert(0, str(resolve(mcfg["code_path"])))
    from brainlm_mae.modeling_brainlm import BrainLMForPretraining

    if args.random_init:
        from brainlm_mae.configuration_brainlm import BrainLMConfig

        torch.manual_seed(args.init_seed)
        m = BrainLMForPretraining(BrainLMConfig.from_pretrained(resolve(mcfg["checkpoint_path"])))
    else:
        m = BrainLMForPretraining.from_pretrained(resolve(mcfg["checkpoint_path"]))
    ratio = float(mcfg["embedding"]["mask_ratio"])
    m.config.mask_ratio = ratio
    m.vit.embeddings.config.mask_ratio = ratio
    enc = m.vit.eval().to(args.device)
    for p in enc.parameters():
        p.requires_grad_(False)

    n_par = int(m.config.num_brain_voxels)
    n_time = int(mcfg["input"]["window"])
    n_patch = n_time // int(m.config.timepoint_patching_size)
    n_tok = n_par * n_patch
    gain = float(mcfg["input"]["gain"])

    xyz = np.loadtxt(resolve(mcfg["input"]["coordinates"]))[:, 1:].astype(np.float32)
    if mcfg["input"]["coordinate_units"] == "unit":
        xyz = (xyz - xyz.min(0)) / (xyz.max(0) - xyz.min(0))
    xyz1 = torch.from_numpy(xyz).unsqueeze(0).to(args.device)
    noise = (torch.arange(n_tok, device=args.device).float() / n_tok).unsqueeze(0)

    lab = pd.read_csv(resolve("external/atlases/A424/A424_Labels_AA-AAc_main_maps.csv"))
    lab.columns = [c.strip().lstrip("﻿") for c in lab.columns]
    net_of_parcel = lab.iloc[:, 3].values.astype(int)      # AA-7, 1-based

    ts_dir = resolve(cfg["processed_path"]) / "timeseries"
    subs = [l.strip() for l in open(resolve(cfg["cohort"]["subject_list"])) if l.strip()]
    if args.subjects_file:
        allow = {l.strip() for l in open(args.subjects_file) if l.strip()}
        subs = [s for s in subs if s in allow]
        print(f"subject filter: {len(subs)} of the cohort, from {args.subjects_file.name}")
    rng = np.random.default_rng(args.seed)

    # Enumerate all (subject, run, window) then sample windows, so the subsample is
    # spread across subjects rather than concentrated in the first few.
    index = [(s, r, w) for s in subs for r in RUNS for w in range(6)
             if (ts_dir / f"{s}_{r}.npy").exists()]
    pick = rng.choice(len(index), size=min(args.n_windows, len(index)), replace=False)
    picked = [index[i] for i in sorted(pick)]
    print(f"{len(picked)} windows x {args.tokens_per_window} tokens "
          f"= {len(picked)*args.tokens_per_window:,} activations")

    acts, meta = [], []
    cache_key, cache = None, None
    for i, (s, r, w) in enumerate(picked, 1):
        if cache_key != (s, r):
            cache = np.load(ts_dir / f"{s}_{r}.npy").astype(np.float32)
            cache_key = (s, r)
        seg = cache[:, w * n_time:(w + 1) * n_time]
        if seg.shape[1] < n_time:
            continue
        sig = torch.from_numpy(seg * gain).unsqueeze(0).to(args.device)
        with torch.no_grad():
            h = enc(signal_vectors=sig, xyz_vectors=xyz1, noise=noise,
                    return_dict=True).last_hidden_state[0, 1:]   # drop CLS
        sel = rng.choice(n_tok, size=args.tokens_per_window, replace=False)
        acts.append(h[sel].cpu().numpy().astype(np.float32))
        parcel = sel // n_patch
        meta.append(np.stack([
            np.full(len(sel), i - 1),                 # window row id
            sel,                                      # token index
            parcel,                                   # parcel, 0-based
            net_of_parcel[parcel],                    # AA-7 network, 1-based
            sel % n_patch,                            # temporal patch
        ], 1))
        if i % 250 == 0 or i == len(picked):
            print(f"  {i}/{len(picked)}", flush=True)

    A = np.concatenate(acts)
    M = np.concatenate(meta)
    win = pd.DataFrame(picked, columns=["subject", "run", "window"])

    name = "token_activations_random_init.npz" if args.random_init else "token_activations.npz"
    out = args.out or (ROOT / "results" / args.dataset / args.model /
                       "interpretability" / name)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, acts=A, meta=M,
             meta_cols=np.array(["window_row", "token", "parcel", "net7", "patch"]),
             subject=win["subject"].values, run=win["run"].values,
             window=win["window"].values)
    print(f"\nactivations {A.shape} {A.dtype} | {A.nbytes/1e9:.2f} GB")
    print(f"mean {A.mean():.4f} sd {A.std():.4f} | frac exactly zero {float((A==0).mean()):.4f}")
    stamp(out.parent, "src/04_interpretability/extract_token_acts_hcp.py", ROOT,
          n_windows=len(picked), tokens_per_window=args.tokens_per_window,
          n_activations=int(A.shape[0]), dim=int(A.shape[1]),
          weights="random_init" if args.random_init else "pretrained",
          init_seed=args.init_seed if args.random_init else None,
          device=args.device)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
