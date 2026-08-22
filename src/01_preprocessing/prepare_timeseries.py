#!/usr/bin/env python3
"""Denoise, scrub-fill, resample and scale parcel time series.

Runs entirely on the cached parcel arrays written by ``parcellate.py``, so it is
cheap to re-run with different settings -- the external drive is never touched.

Pipeline, per subject, in this order:

  1. Drop leading non-steady-state (dummy) volumes.
  2. Nuisance regression at NATIVE TR. Design matrix = 24 head-motion params
     + 8 physiological (csf, white_matter and their expansions) + fMRIPrep's
     DCT cosine basis (the 128 s high-pass) + intercept. No global signal.
     The high-pass enters the SAME design matrix rather than being applied as a
     separate sequential filter, which would otherwise reintroduce variance the
     nuisance regression removed.
     Betas are fit on RETAINED frames only, then applied to all frames: confound
     values at motion-flagged frames are themselves corrupted, so letting them
     into the fit propagates the artifact into every parcel's betas.
  3. Scrub-fill flagged frames by cubic spline over the retained frames. Filling
     rather than excising, because excision breaks the temporal continuity
     BrainLM's positional embeddings assume. This step must be an interpolator:
     removing frames leaves a NON-uniform grid, which polyphase resampling
     cannot accept.
  4. Resample to a common TR with scipy.signal.resample_poly (polyphase FIR).
     Band-limited by construction and non-periodic, so it neither aliases nor
     rings at the run edges. The up/down ratio is source_tr/target_tr,
     rational-approximated under a denominator cap because exact ratios can be
     enormous (TR 2.998 -> 2998/735) and the internal FIR length scales with
     max(up, down). The achieved effective TR is reported.
  5. Robust-scale per parcel (subtract median, divide by IQR), matching
     BrainLM's pretraining normalisation. Applied last, after resampling.

Output: ``<processed_path>/timeseries/<subject_id>.npy``, float32, shape
(n_parcels, n_resampled_timepoints), plus ``prepare_report.csv``.

Usage:
    python src/01_preprocessing/prepare_timeseries.py --dataset hcp_ya --site MI --tr 0.85
    python src/01_preprocessing/prepare_timeseries.py --dataset hcp_ya
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    censor_mask,
    expand_confounds,
    load_config,
    load_confounds,
    resolve,
    subject_files,
)


def build_design_matrix(conf: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Assemble the fixed-width nuisance design matrix.

    Fixed-width is the point: CompCor column counts vary from 1 to 188 across
    subjects in a multi-site cohort, so any CompCor-based strategy would make the
    number of regressors a per-subject processing degree of freedom.
    """
    dcfg = cfg["denoise"]
    blocks, names = [], []

    motion, motion_names = expand_confounds(
        conf, dcfg["motion_params"], dcfg["motion_expansions"]
    )
    blocks.append(motion)
    names += motion_names

    physio, physio_names = expand_confounds(
        conf, dcfg["physio_params"], dcfg["physio_expansions"]
    )
    blocks.append(physio)
    names += physio_names

    if dcfg.get("include_cosine", True):
        cos_cols = sorted(c for c in conf.columns if c.startswith("cosine"))
        if cos_cols:
            blocks.append(conf[cos_cols].to_numpy(dtype=np.float64))
            names += cos_cols

    if dcfg.get("include_non_steady_state", True):
        nss_cols = sorted(c for c in conf.columns if c.startswith("non_steady_state_outlier"))
        if nss_cols:
            blocks.append(conf[nss_cols].to_numpy(dtype=np.float64))
            names += nss_cols

    design = np.column_stack(blocks)
    if dcfg.get("add_intercept", True):
        design = np.column_stack([np.ones(len(design)), design])
        names = ["intercept"] + names
    return design, names


def regress_out(ts: np.ndarray, design: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Residualise ``ts`` (n_parcels, n_time) against ``design``, fitting on ``fit_mask``."""
    y_fit = ts[:, fit_mask].T          # (n_fit, n_parcels)
    x_fit = design[fit_mask]           # (n_fit, n_regressors)
    betas, *_ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
    return (ts.T - design @ betas).T


def scrub_fill(ts: np.ndarray, flagged: np.ndarray, method: str = "cubic") -> np.ndarray:
    """Replace flagged frames by interpolation over the retained frames."""
    if not flagged.any():
        return ts
    retained = ~flagged
    t = np.arange(ts.shape[1], dtype=np.float64)
    out = ts.copy()
    if method == "cubic" and retained.sum() >= 4:
        spline = CubicSpline(t[retained], ts[:, retained], axis=1, extrapolate=True)
        out[:, flagged] = spline(t[flagged])
    else:
        for p in range(ts.shape[0]):
            out[p, flagged] = np.interp(t[flagged], t[retained], ts[p, retained])
    return out


def resample_to_tr(
    ts: np.ndarray, source_tr: float, target_tr: float, cfg: dict
) -> tuple[np.ndarray, float, int, int]:
    """Polyphase-resample from ``source_tr`` to ``target_tr``.

    Returns (resampled, effective_tr, up, down). ``effective_tr`` is what the
    rational approximation actually achieves and is recorded so any drift from
    the requested target stays auditable.
    """
    rcfg = cfg["resample"]
    ratio = Fraction(source_tr / target_tr).limit_denominator(
        int(rcfg.get("max_denominator", 500))
    )
    up, down = ratio.numerator, ratio.denominator
    if up == down:
        return ts, source_tr, up, down

    out = resample_poly(
        ts, up, down, axis=1, padtype=rcfg.get("padtype", "line")
    ).astype(np.float32)
    effective_tr = source_tr * down / up
    return out, effective_tr, up, down


def robust_scale(ts: np.ndarray, iqr_range: list[int]) -> np.ndarray:
    """Per-parcel median/IQR scaling.

    NOTE this does NOT reproduce BrainLM's pretraining normalisation, contrary to
    what this docstring claimed before 2026-08-15. The pretraining column
    (``Voxelwise_RobustScaler_Normalized_Recording``) mean-centres per parcel and
    then subtracts a per-parcel constant; it never divides. See
    ``docs/neurips_workshop_paper_plan.md`` section 8.1.1.

    Two consequences: the output amplitude is ~40x below pretraining, and the
    amplitude differences BETWEEN parcels are removed. ``center_only`` keeps them.
    Which is better is settled empirically by reconstruction R^2, not by argument
    -- ``src/02_finetuning/check_input_scaling.py``.
    """
    lo, hi = iqr_range
    median = np.nanmedian(ts, axis=1, keepdims=True)
    iqr = np.nanpercentile(ts, hi, axis=1, keepdims=True) - np.nanpercentile(
        ts, lo, axis=1, keepdims=True
    )
    # Flat parcels (IQR 0) would divide by zero; leave them centred at 0.
    iqr = np.where(iqr <= 0, 1.0, iqr)
    return ((ts - median) / iqr).astype(np.float32)


def center_only(ts: np.ndarray) -> np.ndarray:
    """Per-parcel mean removal, with no division.

    Keeps the amplitude differences between parcels that ``robust_scale`` removes,
    and leaves the signal at the raw parcel-mean amplitude -- both closer to
    BrainLM's pretraining distribution in shape.
    """
    return (ts - np.nanmean(ts, axis=1, keepdims=True)).astype(np.float32)


def apply_scaling(ts: np.ndarray, cfg: dict) -> np.ndarray:
    """Dispatch on ``scaling.method``."""
    method = cfg["scaling"]["method"]
    if method == "robust":
        return robust_scale(ts, cfg["scaling"]["iqr_range"])
    if method == "center":
        return center_only(ts)
    raise ValueError(f"unknown scaling.method {method!r} (expected 'robust' or 'center')")


def process_subject(cfg: dict, subject_id: str, raw_ts: np.ndarray, tr: float) -> tuple[np.ndarray, dict]:
    """Run the full per-subject pipeline. Returns (timeseries, report row)."""
    _, conf_path = subject_files(cfg, subject_id)
    conf = load_confounds(conf_path)

    if len(conf) != raw_ts.shape[1]:
        raise ValueError(
            f"confounds rows ({len(conf)}) != parcel timepoints ({raw_ts.shape[1]})"
        )

    # Parcels with no voxels on this subject's grid come back all-NaN. The
    # reference A424 implementation (Get_A424_TS.m) zero-fills them, and NaN
    # would otherwise propagate through the regression. Zeroed BEFORE denoising
    # so the design matrix never sees NaN.
    raw_ts = raw_ts.astype(np.float64).copy()
    empty = np.isnan(raw_ts).all(axis=1)
    n_zerofilled = int(empty.sum())
    raw_ts[empty] = 0.0

    # Parcels zeroed for every subject regardless of recoverability, because
    # their availability is site-linked and would leak scanner identity.
    forced = [p - 1 for p in (cfg["atlas"].get("zero_parcels") or [])]
    if forced:
        raw_ts[forced] = 0.0

    flagged = censor_mask(conf, cfg["denoise"]["censor_column_prefix"])

    # 1. Drop dummy scans, from both the signal and the confounds.
    nss_cols = [c for c in conf.columns if c.startswith("non_steady_state_outlier")]
    n_drop = 0
    if nss_cols:
        nss = conf[nss_cols].to_numpy().sum(axis=1) > 0
        # Dummies are leading frames; count the initial run only.
        while n_drop < len(nss) and nss[n_drop]:
            n_drop += 1
    ts = raw_ts[:, n_drop:]
    conf = conf.iloc[n_drop:].reset_index(drop=True)
    flagged = flagged[n_drop:]

    # 2. Nuisance regression at native TR, fit on retained frames only.
    design, reg_names = build_design_matrix(conf, cfg)
    retained = ~flagged
    if retained.sum() <= design.shape[1]:
        raise ValueError(
            f"only {int(retained.sum())} retained frames for {design.shape[1]} regressors"
        )
    ts = regress_out(ts, design, retained)

    # 3. Fill flagged frames.
    ts = scrub_fill(ts, flagged, cfg["denoise"].get("scrub_fill", "cubic"))

    # 4. Resample to the common TR.
    target_tr = float(cfg["resample"]["target_tr"])
    ts, effective_tr, up, down = resample_to_tr(ts, tr, target_tr, cfg)

    # 5. Scale.
    ts = apply_scaling(ts, cfg)

    report = {
        "subject_id": subject_id,
        "source_tr": tr,
        "effective_tr": effective_tr,
        "resample_up": up,
        "resample_down": down,
        "n_dummy_dropped": n_drop,
        "n_regressors": design.shape[1],
        "n_frames_native": int(len(flagged)),
        "n_frames_flagged": int(flagged.sum()),
        "frac_flagged": float(flagged.mean()),
        "n_frames_resampled": int(ts.shape[1]),
        "nyquist_hz": 1.0 / (2.0 * tr),
        "n_parcels_zerofilled": n_zerofilled,
        "n_parcels_forced_zero": len(forced),
        "n_nan_parcels": int(np.isnan(ts).all(axis=1).sum()),
    }
    return ts, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--site", default=None)
    parser.add_argument("--tr", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--scaling",
        choices=["robust", "center"],
        default=None,
        help="override scaling.method for a variant build; config value is used if omitted",
    )
    parser.add_argument(
        "--out-subdir",
        default="timeseries",
        help="write under <processed_path>/<out-subdir>; use a new name for variant builds "
        "so the canonical outputs are never clobbered",
    )
    args = parser.parse_args()

    cfg = load_config(args.dataset)
    if args.scaling:
        cfg["scaling"]["method"] = args.scaling
    interim = resolve(cfg["interim_path"])
    processed = resolve(cfg["processed_path"])

    man = pd.read_csv(interim / "manifest.csv")
    excluded = set(cfg["cohort"].get("exclude_subjects") or [])
    man = man[~man["subject_id"].isin(excluded)]
    if "error" in man:
        man = man[man["error"].isna()]
    if args.site:
        man = man[man["site"] == args.site]
    if args.tr is not None:
        man = man[np.isclose(man["tr"], args.tr, atol=0.01)]
    if args.limit:
        man = man.head(args.limit)

    parcels_dir = interim / "parcels_raw"
    out_dir = processed / args.out_subdir
    # Variant builds must never overwrite the canonical per-subject records. The
    # default subdir keeps the documented paths; anything else is suffixed.
    rep_suffix = "" if args.out_subdir == "timeseries" else f"_{args.out_subdir}"
    out_dir.mkdir(parents=True, exist_ok=True)

    floor = int(cfg["cohort"]["min_usable_frames"])
    print(
        f"Preparing {len(man)} subjects -> target TR {cfg['resample']['target_tr']}s, "
        f"scaling {cfg['scaling']['method']} -> {out_dir}"
    )

    reports, skipped = [], []
    for i, row in enumerate(man.itertuples(), 1):
        out_path = out_dir / f"{row.subject_id}.npy"
        if out_path.exists() and not args.overwrite:
            continue
        raw_path = parcels_dir / f"{row.subject_id}.npy"
        if not raw_path.exists():
            skipped.append({"subject_id": row.subject_id, "reason": "no parcel file"})
            continue
        try:
            ts, rep = process_subject(cfg, row.subject_id, np.load(raw_path), float(row.tr))
        except Exception as exc:
            skipped.append({"subject_id": row.subject_id, "reason": str(exc)})
            continue

        # Usable-frame floor is applied on NATIVE frames. Resampling inflates the
        # timepoint count without adding information, so judging sufficiency on
        # the resampled length would let a 120-volume run masquerade as a long one.
        native_usable = rep["n_frames_native"] - rep["n_frames_flagged"]
        rep["native_usable_frames"] = native_usable
        rep["meets_frame_floor"] = native_usable >= floor

        np.save(out_path, ts)
        reports.append(rep)
        if i % 25 == 0 or i == len(man):
            print(f"  {i}/{len(man)}")

    if reports:
        rep_df = pd.DataFrame(reports)
        rep_path = processed / f"prepare_report{rep_suffix}.csv"
        rep_df.to_csv(rep_path, index=False)
        print(f"\nWrote {len(rep_df)} subjects -> {out_dir}")
        print(f"Report -> {rep_path}")
        print(f"  meeting the {floor}-native-frame floor: "
              f"{int(rep_df['meets_frame_floor'].sum())}/{len(rep_df)}")
        drift = (rep_df["effective_tr"] - float(cfg["resample"]["target_tr"])).abs().max()
        print(f"  max effective-TR drift from target: {drift:.6f}s")
    if skipped:
        skip_df = pd.DataFrame(skipped)
        skip_path = processed / f"prepare_skipped{rep_suffix}.csv"
        skip_df.to_csv(skip_path, index=False)
        print(f"  skipped {len(skip_df)} -> {skip_path}")


if __name__ == "__main__":
    main()
