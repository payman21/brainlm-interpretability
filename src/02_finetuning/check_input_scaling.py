#!/usr/bin/env python3
"""Day-1 input format check for BrainLM on A424 parcel time series.

WHY THIS EXISTS
---------------
BrainLM was pretrained on parcel time series at the raw UK Biobank amplitude
(the toolkit records a global std of ~41.4). This project's ``.npy`` files are
robust-scaled per subject per parcel, so they sit at median 0 / IQR 1, with a
measured std near 0.9 -- roughly 40x smaller. A masked autoencoder is not
invariant to input scale, so feeding it data at the wrong amplitude can drive
reconstruction to zero and produce a false "BrainLM does not transfer" result.

That failure mode is dangerous here specifically because "BrainLM does not
transfer" is a conclusion the project might otherwise want to draw. A
self-inflicted scaling error would be indistinguishable from a real finding.

Which normalisation trained the released ``old_13M`` checkpoint is not
recorded (it predates the toolkit's current code, and carries mask_ratio 0.2
against 0.75 in the later checkpoints). So this is settled by measurement, not
by reading code.

WHAT IT DOES
------------
Runs masked reconstruction on a handful of subjects across a grid of

  * input gains        -- the multiplier applied to the signal
  * coordinate scalings -- raw MNI mm, or min-max normalised to [0, 1]

and reports R^2 on the masked elements only.

R^2 is the right metric here and the model's own MSE loss is not: multiplying
the input by k scales MSE by k^2, so losses are not comparable across gains.
R^2 normalises by the variance of the masked ground truth, which scales the
same way, so it is.

The mask is held identical across every cell of the grid (one fixed noise
tensor per subject), so cells differ only by the thing being tested.

Usage:
    python src/02_finetuning/check_input_scaling.py --dataset hcp_ya
    python src/02_finetuning/check_input_scaling.py --dataset hcp_ya --n-subjects 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
sys.path.insert(0, str(ROOT / "models" / "foundation" / "brainlm" / "src"))

from common import load_config, resolve  # noqa: E402


def load_coords(path: Path, n_parcels: int, mode: str) -> torch.Tensor:
    """A424 parcel centroids as [n_parcels, 3].

    The file ships with BrainLM (``toolkit/atlases/A424_Coordinates.dat``) and is
    label, X, Y, Z in MNI mm, one row per parcel in ascending label order -- the
    same order ``parcellate.py`` writes. ``mode`` selects raw mm or a min-max
    normalisation to [0, 1]; which one pretraining used is not documented, so
    both are on the grid.
    """
    arr = np.loadtxt(path)
    if arr.shape != (n_parcels, 4):
        raise ValueError(f"expected ({n_parcels}, 4) in {path}, got {arr.shape}")
    xyz = arr[:, 1:].astype(np.float32)
    if mode == "unit":
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        xyz = (xyz - lo) / (hi - lo)
    elif mode != "mm":
        raise ValueError(f"unknown coord mode {mode}")
    return torch.from_numpy(xyz)


def centre_window(ts: np.ndarray, n_timepoints: int) -> np.ndarray:
    """The middle ``n_timepoints`` frames, to avoid run-edge effects."""
    if ts.shape[1] < n_timepoints:
        raise ValueError(f"only {ts.shape[1]} frames, need {n_timepoints}")
    start = (ts.shape[1] - n_timepoints) // 2
    return ts[:, start : start + n_timepoints]


def masked_r2(true: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> float:
    """R^2 over masked elements only. mask is 1 where the model had to predict."""
    m = mask.unsqueeze(-1).expand_as(pred).bool()
    t, p = true[m], pred[m]
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--site", default="MI")
    ap.add_argument("--tr", type=float, default=0.85)
    ap.add_argument("--n-subjects", type=int, default=5)
    ap.add_argument("--gains", type=float, nargs="+", default=[1.0, 10.0, 40.0, 100.0])
    ap.add_argument("--coord-modes", nargs="+", default=["mm", "unit"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or mps; cpu is fast enough here")
    ap.add_argument(
        "--subdir",
        default="timeseries",
        help="input dir under <processed_path>; use a variant build to compare scalings",
    )
    ap.add_argument(
        "--checkpoint",
        default="models/foundation/brainlm/checkpoints/old_13M",
        help="old_13M is the checkpoint whose config matches this codebase and the A424 input",
    )
    args = ap.parse_args()

    cfg = load_config(args.dataset)

    from brainlm_mae.modeling_brainlm import BrainLMForPretraining

    model = BrainLMForPretraining.from_pretrained(resolve(args.checkpoint))
    model.eval().to(args.device)
    n_parcels = model.config.num_brain_voxels
    n_timepoints = model.config.num_timepoints_per_voxel
    n_tokens = n_timepoints // model.config.timepoint_patching_size
    print(
        f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
        f"{n_parcels} parcels x {n_timepoints} timepoints, "
        f"{n_tokens} tokens/parcel, mask_ratio {model.config.mask_ratio}"
    )

    coords = {
        m: load_coords(
            resolve("external/atlases/A424/A424_Coordinates.dat"), n_parcels, m
        ).to(args.device)
        for m in args.coord_modes
    }

    interim, processed = resolve(cfg["interim_path"]), resolve(cfg["processed_path"])
    ts_dir = processed / args.subdir

    # Subject selection differs by dataset. A multi-site cohort needs a single
    # site/TR stratum picked out of the manifest, because its TRs and sites vary
    # and the sweep must not mix them. HCP is single-site single-protocol with one file per
    # run, so there is no stratum to select and the manifest does not exist --
    # just take files off disk.
    manifest = interim / "manifest.csv"
    if manifest.exists():
        man = pd.read_csv(manifest)
        man = man[~man["subject_id"].isin(set(cfg["cohort"].get("exclude_subjects") or []))]
        man = man[(man["site"] == args.site) & (np.isclose(man["tr"], args.tr, atol=0.01))]
        candidates = list(man["subject_id"])
        print(f"selection: manifest, site={args.site} tr={args.tr} -> {len(candidates)}")
    else:
        candidates = sorted(f.stem for f in ts_dir.glob("*.npy"))
        print(f"selection: all files under {ts_dir} -> {len(candidates)}")

    rows, used = [], []
    for sid in candidates:
        path = ts_dir / f"{sid}.npy"
        if not path.exists():
            continue
        ts = np.load(path)
        if ts.shape[1] < n_timepoints or not np.isfinite(ts).all():
            continue

        window = torch.from_numpy(centre_window(ts, n_timepoints)).float()
        signal = window.unsqueeze(0).to(args.device)  # [1, n_parcels, n_timepoints]

        # One fixed noise vector per subject fixes the mask across the whole
        # grid, so cells differ only by gain and coordinate scaling.
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        noise = torch.rand(1, n_parcels * n_tokens, generator=gen).to(args.device)

        for cmode in args.coord_modes:
            xyz = coords[cmode].unsqueeze(0)
            for gain in args.gains:
                with torch.no_grad():
                    out = model(
                        signal_vectors=signal * gain,
                        xyz_vectors=xyz,
                        noise=noise,
                        return_dict=True,
                    )
                pred = out.logits[0]  # (logits, latent)
                true = (signal * gain).reshape(pred.shape)
                rows.append(
                    {
                        "subject_id": sid,
                        "coord_mode": cmode,
                        "gain": gain,
                        "r2": masked_r2(true, pred, out.mask.reshape(pred.shape[:-1])),
                    }
                )
        used.append(sid)
        print(f"  {len(used)}/{args.n_subjects}  {sid}")
        if len(used) >= args.n_subjects:
            break

    if not rows:
        sys.exit(f"No usable inputs found under {ts_dir}.")

    df = pd.DataFrame(rows)
    grid = (
        df.groupby(["coord_mode", "gain"])["r2"]
        .agg(r2_mean="mean", r2_sd="std")
        .reset_index()
    )

    print(f"\nMasked reconstruction R^2 over {len(used)} subjects")
    print("(BrainLM reports ~0.40 on held-out UKB and ~0.32 on HCP as an external cohort)\n")
    print(f"{'coords':>8} {'gain':>8} {'mean R2':>10} {'sd':>8}")
    for r in grid.itertuples():
        print(f"{r.coord_mode:>8} {r.gain:>8.3g} {r.r2_mean:>10.4f} {r.r2_sd:>8.4f}")

    best = grid.loc[grid["r2_mean"].idxmax()]
    spread = grid["r2_mean"].max() - grid["r2_mean"].min()

    print(f"\nbest: coords={best['coord_mode']} gain={best['gain']:.0f} "
          f"R2={best['r2_mean']:.4f}   spread across grid={spread:.4f}")
    print("\nVERDICT")
    if best["r2_mean"] < 0.05:
        print("  Every cell is near zero. The problem is NOT input scaling.")
        print("  Check parcel order, the xyz array, and the checkpoint before")
        print("  touching normalisation. Do not tune scaling to rescue another bug.")
    elif spread < 0.02:
        print("  Flat across the grid. Input scale does not matter for this model.")
        print("  Keep the current robust-scaled files unchanged. Record this in the")
        print("  paper appendix as a one-line negative control and move on.")
    else:
        print(f"  Clear optimum at coords={best['coord_mode']}, gain={best['gain']:.3g}.")
        print("  Apply that gain to the model input. Do NOT re-tune it later, and")
        print("  never re-select it on diagnosis accuracy -- this choice must stay")
        print("  label-free so it cannot leak into the classification result.")

    dataset_dir = args.dataset
    out_dir = ROOT / "results" / dataset_dir / "brainlm" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"input_scaling_check_{args.subdir}.csv", index=False)
    print(f"\nper-subject results -> {out_dir / f'input_scaling_check_{args.subdir}.csv'}")


if __name__ == "__main__":
    main()
