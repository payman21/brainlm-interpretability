#!/usr/bin/env python3
"""Build the per-subject QC manifest for a dataset.

Scans the raw fMRIPrep derivatives once and writes one row per subject to
``<interim_path>/manifest.csv``. Every dataset attribute quoted downstream comes
from this table, so it is the reproducible replacement for ad-hoc audit scripts.

Reads only NIfTI headers, JSON sidecars and confound TSVs -- the BOLD voxel data
is never loaded, so this runs in a few minutes over the full cohort rather than
streaming 402 GB.

Columns written:
    subject_id, site, tr, tr_header, n_vol, n_vol_header, duration_min,
    nyquist_hz, slice_timing_corrected, mean_fd, max_fd, mean_dvars,
    mean_global_signal, n_motion_outliers, frac_flagged, n_usable_frames,
    n_non_steady_state, n_confound_cols, n_compcor_cols, n_cosine_cols,
    grid_shape, voxel_size, tr_mismatch, bold_path, confounds_path

Usage:
    python src/01_preprocessing/build_manifest.py --dataset hcp_ya
    python src/01_preprocessing/build_manifest.py --dataset hcp_ya --limit 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    count_prefix,
    censor_mask,
    find_subjects,
    load_config,
    load_confounds,
    read_bold_sidecar,
    resolve,
    site_of,
    subject_files,
)


def scan_subject(cfg: dict, subject_id: str) -> dict:
    """Collect every header-level and confound-level QC quantity for one subject."""
    row: dict = {"subject_id": subject_id, "site": site_of(subject_id)}

    bold_path, conf_path = subject_files(cfg, subject_id)
    row["bold_path"] = str(bold_path) if bold_path else ""
    row["confounds_path"] = str(conf_path) if conf_path else ""
    if bold_path is None or conf_path is None:
        row["error"] = "missing bold or confounds file"
        return row

    # --- JSON sidecar -----------------------------------------------------
    sidecar = read_bold_sidecar(bold_path)
    tr = sidecar.get("RepetitionTime")
    row["tr"] = float(tr) if tr is not None else np.nan
    # Not uniform across sites: 977 subjects corrected, 486 not, split by site
    # and in two sites split *within* site. Carried as a covariate.
    row["slice_timing_corrected"] = sidecar.get("SliceTimingCorrected", None)

    # --- NIfTI header (no voxel data loaded) ------------------------------
    img = nib.load(bold_path)
    shape = img.header.get_data_shape()
    zooms = img.header.get_zooms()
    row["n_vol_header"] = int(shape[3]) if len(shape) > 3 else np.nan
    row["tr_header"] = float(zooms[3]) if len(zooms) > 3 else np.nan
    row["grid_shape"] = "x".join(str(s) for s in shape[:3])
    row["voxel_size"] = "x".join(f"{float(z):.2f}" for z in zooms[:3])

    # --- confounds --------------------------------------------------------
    conf = load_confounds(conf_path)
    n_vol = len(conf)
    row["n_vol"] = n_vol
    row["duration_min"] = (row["tr"] * n_vol / 60.0) if np.isfinite(row["tr"]) else np.nan
    # Source Nyquist. Resampling to a finer grid does not raise this -- it is the
    # spectral ceiling the subject carries into the model, and is site-linked.
    row["nyquist_hz"] = (1.0 / (2.0 * row["tr"])) if np.isfinite(row["tr"]) else np.nan

    if "framewise_displacement" in conf:
        fd = conf["framewise_displacement"].to_numpy()
        row["mean_fd"] = float(np.mean(fd))
        row["max_fd"] = float(np.max(fd))
    if "std_dvars" in conf:
        row["mean_dvars"] = float(np.mean(conf["std_dvars"].to_numpy()))
    # Recorded for QC only. Global signal is NOT regressed out (see config).
    if "global_signal" in conf:
        row["mean_global_signal"] = float(np.mean(conf["global_signal"].to_numpy()))

    flagged = censor_mask(conf, cfg["denoise"]["censor_column_prefix"])
    row["n_motion_outliers"] = int(flagged.sum())
    row["frac_flagged"] = float(flagged.mean()) if n_vol else np.nan

    n_nss = count_prefix(conf, "non_steady_state_outlier")
    row["n_non_steady_state"] = n_nss
    # Frames left after dropping dummy scans and scrubbed frames. This is the
    # number that decides whether a subject can fill BrainLM's 200-timestep
    # window, not n_vol.
    row["n_usable_frames"] = int(n_vol - n_nss - flagged.sum())

    row["n_confound_cols"] = len(conf.columns)
    row["n_compcor_cols"] = sum(1 for c in conf.columns if "comp_cor" in c)
    row["n_cosine_cols"] = count_prefix(conf, "cosine")

    # Sidecar vs header TR disagreement -- 8 subjects in one multi-site cohort.
    row["tr_mismatch"] = bool(
        np.isfinite(row["tr"])
        and np.isfinite(row["tr_header"])
        and abs(row["tr"] - row["tr_header"]) > 0.02
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="config name, e.g. hcp_ya")
    parser.add_argument("--limit", type=int, default=None, help="scan first N subjects only")
    parser.add_argument("--output", default=None, help="override output path")
    args = parser.parse_args()

    cfg = load_config(args.dataset)
    subjects = find_subjects(cfg)
    if args.limit:
        subjects = subjects[: args.limit]

    print(f"Scanning {len(subjects)} subjects from {resolve(cfg['raw_path'])}")
    rows = []
    for i, subject_id in enumerate(subjects, 1):
        try:
            rows.append(scan_subject(cfg, subject_id))
        except Exception as exc:  # keep going; record the failure
            rows.append({"subject_id": subject_id, "site": site_of(subject_id), "error": str(exc)})
        if i % 100 == 0 or i == len(subjects):
            print(f"  {i}/{len(subjects)}")

    df = pd.DataFrame(rows).sort_values("subject_id")

    out = Path(args.output) if args.output else resolve(cfg["interim_path"]) / "manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}")

    # --- summary ----------------------------------------------------------
    errs = df["error"].notna().sum() if "error" in df else 0
    print(f"  errors: {errs}")
    if "tr" in df:
        print(f"  distinct TRs: {df['tr'].round(2).nunique()}   median TR: {df['tr'].median():.2f}")
    if "n_usable_frames" in df:
        floor = cfg["cohort"]["min_usable_frames"]
        ok = (df["n_usable_frames"] >= floor).sum()
        print(f"  subjects with >= {floor} usable frames: {ok}/{len(df)}")
    if "tr_mismatch" in df:
        print(f"  TR sidecar/header mismatches: {int(df['tr_mismatch'].sum())}")


if __name__ == "__main__":
    main()
