#!/usr/bin/env python3
"""Step 2 -- per-window mean SAE feature activations over ALL tokens.

Why this exists
---------------
The attribution in sae_attribution_hcp.py needs, for every window, each SAE feature's
MEAN activation over that window's 4,240 tokens. `sae_features.npz` cannot supply it:
token_activations.npz sampled 100 tokens from each of 5,000 windows, so a subject-level
mean taken from it averages ~540 tokens out of ~102,000 -- a 0.5% sample. For features
that fire rarely that estimate is mostly noise, and because the attribution ranks
features by how much their contribution VARIES across subjects, sampling noise inflates
exactly the statistic being ranked on. The ranking would partly be a ranking of which
features were sampled worst.

So this recomputes from scratch: BrainLM forward over every window, SAE-encode all
4,240 tokens, keep only the per-window mean. Nothing token-level is stored -- the full
tensor would be 22,482 x 4,240 x 4,096 -- so the output is 22,482 x width, ~370 MB.

Two things it stores beyond the feature means
---------------------------------------------
`emb_mean` is the token-mean embedding computed in the SAME forward pass. The
downstream readout is a ridge on exactly that vector, so taking it from here rather
than from the committed frozen_embeddings.npz keeps the readout weights and the feature
activations on one device with one set of numerics. BrainLM's encoder is order-sensitive
(Nystromformer landmarks are means over consecutive tokens) and CPU/MPS/CUDA differ at
~1e-5; pairing a CUDA re-encode against CPU-era embeddings would silently mix them.

It is also a check. The script reports the largest absolute difference against
frozen_embeddings.npz on the shared keys. That number should be ~1e-5 or smaller. If it
is large, the forward pass here is not configured the way the embeddings were and
nothing downstream is trustworthy.

The frozen threshold
--------------------
Encoding uses `encode_fixed` with the `thresh` constant stored in sae.pt, never
`encode`. BatchTopK's cut is a property of whatever batch it was handed, so calling
`encode` here -- one window of 4,240 tokens at a time -- would apply a different,
per-window sparsity cut than the one the dictionary was trained under. If sae.pt has no
`thresh` key it predates that fix and must be retrained.

Usage:
    python src/04_interpretability/encode_sae_windows_hcp.py --device mps
    python src/04_interpretability/encode_sae_windows_hcp.py --device mps --limit 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "04_interpretability"))
from common import load_config, resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402
from train_sae_hcp import BatchTopKSAE  # noqa: E402

RUNS = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]


def identity_noise(n_tokens: int, device: str) -> torch.Tensor:
    """Monotonic noise so BrainLM's argsort-based shuffle is the identity.

    Same helper as extract_embeddings.py, and required for the same reason: BrainLM's
    random_masking() draws torch.rand and argsorts it on every call, permuting all
    tokens even at mask_ratio 0 where it drops none. The encoder is order-sensitive, so
    left alone the same window gives a different embedding on every call.
    """
    return (torch.arange(n_tokens, dtype=torch.float32, device=device) / n_tokens).unsqueeze(0)


def load_coords(path: Path, n_parcels: int, units: str) -> torch.Tensor:
    arr = np.loadtxt(path)
    if arr.shape != (n_parcels, 4):
        raise ValueError(f"expected ({n_parcels}, 4) in {path}, got {arr.shape}")
    if not (arr[:, 0] == np.arange(1, n_parcels + 1)).all():
        raise ValueError(f"{path} is not in ascending label order")
    xyz = arr[:, 1:].astype(np.float32)
    if units == "unit":
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        xyz = (xyz - lo) / (hi - lo)
    elif units != "mm":
        raise ValueError(f"unknown coordinate_units {units!r}")
    return torch.from_numpy(xyz)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--model", default="brainlm")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="first N subjects, for smoke tests")
    ap.add_argument("--sae", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    with (ROOT / "configs" / "models" / f"{args.model}.yaml").open() as fh:
        mcfg = yaml.safe_load(fh)
    interp = ROOT / "results" / args.dataset / args.model / "interpretability"

    ckpt = torch.load(args.sae or (interp / "sae" / "sae.pt"),
                      map_location="cpu", weights_only=False)
    if "thresh" not in ckpt:
        raise SystemExit(
            "sae.pt has no 'thresh' key, so it predates the frozen-threshold fix and its "
            "activations are not reproducible outside the batch they were computed in. "
            "Retrain with train_sae_hcp.py.")
    width, k, thresh = int(ckpt["width"]), int(ckpt["k"]), float(ckpt["thresh"])
    mu = torch.as_tensor(np.asarray(ckpt["mu"]), dtype=torch.float32).to(args.device)
    scale = float(ckpt["scale"])
    sae = BatchTopKSAE(mu.shape[-1], width, k)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval().to(args.device)
    for p in sae.parameters():
        p.requires_grad_(False)
    print(f"SAE: width {width}, k {k}, frozen threshold {thresh:.6g}, scale {scale:.4f}")

    sys.path.insert(0, str(resolve(mcfg["code_path"])))
    from brainlm_mae.modeling_brainlm import BrainLMForPretraining

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
    xyz = load_coords(resolve(mcfg["input"]["coordinates"]), n_par,
                      mcfg["input"]["coordinate_units"]).unsqueeze(0).to(args.device)
    noise = identity_noise(n_tok, args.device)
    print(f"BrainLM: {n_par} parcels x {n_patch} patches = {n_tok} tokens, gain {gain}, "
          f"device {args.device}")

    ts_dir = resolve(cfg["processed_path"]) / "timeseries"
    cohort = [l.strip() for l in open(resolve(cfg["cohort"]["subject_list"])) if l.strip()]
    subs = [s for s in cohort if (ts_dir / f"{s}_{RUNS[0]}.npy").exists()]
    if args.limit:
        subs = subs[: args.limit]

    keys, Zb, Em = [], [], []
    t0 = time.time()
    for i, s in enumerate(subs, 1):
        for r in RUNS:
            p = ts_dir / f"{s}_{r}.npy"
            if not p.exists():
                continue
            a = np.load(p).astype(np.float32)
            for w in range(a.shape[1] // n_time):
                seg = a[:, w * n_time:(w + 1) * n_time]
                if seg.shape[1] < n_time:
                    continue
                sig = torch.from_numpy(np.ascontiguousarray(seg) * gain).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    h = enc(signal_vectors=sig, xyz_vectors=xyz, noise=noise,
                            return_dict=True).last_hidden_state[0, 1:]     # drop CLS
                    z = sae.encode_fixed((h - mu) / scale, thresh)
                    Zb.append(z.mean(0).cpu().numpy())
                    Em.append(h.mean(0).cpu().numpy())
                keys.append((s, r, w))
        if i % 25 == 0 or i == len(subs):
            el = time.time() - t0
            print(f"  {i}/{len(subs)} subjects, {len(keys)} windows, {el:.0f}s "
                  f"(eta {el / i * (len(subs) - i):.0f}s)", flush=True)

    Zbar = np.stack(Zb).astype(np.float32)
    Emean = np.stack(Em).astype(np.float32)
    subject = np.array([k[0] for k in keys])
    run = np.array([k[1] for k in keys])
    window = np.array([k[2] for k in keys], dtype=np.int16)
    alive = (Zbar > 0).any(0)
    print(f"\n{Zbar.shape[0]} windows x {width} features | "
          f"{int(alive.sum())} features nonzero in at least one window")
    print(f"mean L0 of the window-mean vector: {float((Zbar > 0).sum(1).mean()):.0f} "
          f"(a MEAN over 4,240 tokens is dense even when each token is sparse)")

    # The check that the forward pass matches the committed embeddings. Join on keys,
    # never on position -- the two scripts enumerate subjects differently.
    fz = ROOT / f"results/{args.dataset}/{args.model}/embeddings/frozen_embeddings.npz"
    if fz.exists():
        z0 = np.load(fz, allow_pickle=True)
        idx = {(a, b, int(c)): j for j, (a, b, c) in
               enumerate(zip(z0["subject_id"].astype(str), z0["run"].astype(str), z0["window"]))}
        sel = np.array([idx.get(kk, -1) for kk in keys])
        ok = sel >= 0
        if ok.any():
            d = np.abs(np.asarray(z0["mean"])[sel[ok]] - Emean[ok])
            print(f"vs frozen_embeddings.npz on {int(ok.sum())} shared windows: "
                  f"max abs diff {d.max():.3g}, median {np.median(d):.3g}")
            if d.max() > 1e-2:
                print("  WARNING: too large to be device numerics. The forward pass here is "
                      "not configured the way frozen_embeddings.npz was -- check gain, "
                      "coordinate units and mask_ratio before using any of this.")
        else:
            print("no shared keys with frozen_embeddings.npz; skipped the parity check")

    out = args.out or (interp / "sae" / "sae_window_means.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, Zbar=Zbar, emb_mean=Emean, alive=alive,
                        subject=subject, run=run, window=window,
                        thresh=np.float32(thresh), scale=np.float32(scale), mu=np.asarray(ckpt["mu"]))
    stamp(interp / "sae", "src/04_interpretability/encode_sae_windows_hcp.py", ROOT,
          n_windows=int(Zbar.shape[0]), n_subjects=len(set(subject.tolist())),
          width=width, thresh=float(thresh), device=args.device,
          seconds=float(time.time() - t0))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
