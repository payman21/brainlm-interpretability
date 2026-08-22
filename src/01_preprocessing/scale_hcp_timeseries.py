#!/usr/bin/env python3
"""Turn raw HCP-YA A424 parcel means into model-ready timeseries.

This is the HCP counterpart of the scaling stage of ``prepare_timeseries.py``,
and it is deliberately much smaller. That script does nuisance regression,
scrub-filling and polyphase resampling because a multi-site clinical cohort
needs all three. HCP needs
none of them:

  * denoising      -- the source is already ICA-FIX cleaned (``hp2000_clean``),
                      the same denoising family BrainLM pretrained on. There are
                      no fMRIPrep confounds to regress and nothing to add.
  * resampling     -- disabled by config. TR 0.72 s against a pretraining 0.735 s
                      is a 2% offset, uniform across a single-protocol cohort, and
                      not worth an interpolation. See plan section 5.6.
  * scrub-filling  -- no frame censoring is applied; there is no motion-based
                      exclusion in this cohort (plan section 9 item 5).

What is left is the robust scaling, per subject per parcel, matching
``scaling:`` in ``configs/datasets/hcp_ya.yaml``.

Usage:
    python src/01_preprocessing/scale_hcp_timeseries.py --dataset hcp_ya
    python src/01_preprocessing/scale_hcp_timeseries.py --in-dir <dir> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
from common import load_config, resolve  # noqa: E402


def robust_scale(ts: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Subtract the median and divide by the IQR, per parcel.

    Parcels with zero IQR are left at zero rather than divided: an empty parcel
    (e.g. frontopolar dropout) has no scale to normalise to, and BrainLM's
    toolkit treats zero as missing.
    """
    med = np.median(ts, axis=1, keepdims=True)
    iqr = np.subtract(*np.percentile(ts, [hi, lo], axis=1))[:, None]
    out = np.zeros_like(ts, dtype=np.float32)
    ok = (iqr[:, 0] > 0)
    out[ok] = ((ts[ok] - med[ok]) / iqr[ok]).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--in-dir", type=Path, help="override: dir of raw parcel .npy")
    ap.add_argument("--out-dir", type=Path, help="override: destination dir")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    lo, hi = cfg["scaling"]["iqr_range"]
    zero_parcels = cfg["atlas"].get("zero_parcels") or []

    in_dir = args.in_dir or resolve(cfg["interim_path"]) / "parcels_raw"
    out_dir = args.out_dir or resolve(cfg["processed_path"]) / "timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.npy"))
    if not files:
        sys.exit(f"No parcel files in {in_dir}")

    print(f"{len(files)} files: {in_dir} -> {out_dir}")
    print(f"robust scale, IQR [{lo}, {hi}], per subject per parcel; "
          f"zeroing parcels {zero_parcels}")

    n = 0
    for f in files:
        dest = out_dir / f.name
        if dest.exists() and not args.overwrite:
            continue
        ts = np.load(f).astype(np.float32)
        out = robust_scale(ts, lo, hi)
        for p in zero_parcels:
            out[p - 1] = 0.0                      # labels are 1-based
        if not np.isfinite(out).all():
            sys.exit(f"non-finite output for {f.name}")
        np.save(dest, out)
        n += 1

    print(f"wrote {n} files")
    if n:
        a = np.load(out_dir / files[0].name)
        print(f"example {files[0].name}: shape {a.shape} "
              f"mean {a.mean():.4f} std {a.std():.4f}")


if __name__ == "__main__":
    main()
