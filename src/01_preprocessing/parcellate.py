#!/usr/bin/env python3
"""Extract parcel time series from preprocessed BOLD volumes.

This is the ONLY step that touches voxel data, and it is what makes everything
downstream cheap: 402 GB of 4D NIfTI becomes ~400 KB per subject
(n_parcels x n_timepoints, float32), so the whole cohort fits in well under a
gigabyte and every later experiment runs off that instead of the external drive.

Denoising and TR resampling are deliberately NOT done here. Nuisance regression,
scrub-filling and polyphase resampling are all linear operations along time and
identical across voxels, so they commute with the spatial mean that parcellation
performs -- applying them to 424 parcel time series gives the same answer as
applying them to ~200,000 voxels, for ~1/500th the compute and storage. They live
in ``prepare_timeseries.py``, which can therefore be re-run with different
settings without re-reading the drive.

The atlas is resampled to each subject's native grid (nearest-neighbour) rather
than resampling BOLD onto a common grid, which would put an extra interpolation
on the signal itself. Note that 13 distinct voxel grids exist across sites, with
slice thickness from 2.2 mm to 5.0 mm, so parcel-mean smoothing differs by site
regardless of what is done here.

Output: ``<interim_path>/parcels_raw/<subject_id>.npy``, shape (n_parcels, n_vol),
float32, in ascending parcel-label order. Parcels with no voxels in a subject's
field of view are written as NaN and recorded in the coverage report.

Usage:
    python src/01_preprocessing/parcellate.py --dataset hcp_ya --site MI --tr 0.85
    python src/01_preprocessing/parcellate.py --dataset hcp_ya
    python src/01_preprocessing/parcellate.py --dataset hcp_ya --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve, subject_files  # noqa: E402


def warp_atlas_to_bold_space(
    atlas_img: nib.Nifti1Image, atlas_space: str, bold_space: str
) -> tuple[nib.Nifti1Image, str]:
    """Warp a label image between MNI template variants.

    A424 ships in MNI152NLin6Asym (the FSL MNI152 grid), while fMRIPrep
    derivatives are usually MNI152NLin2009cAsym. These differ nonlinearly by a few
    millimetres, mostly near cortical edges and inferior structures. Aligning by
    affine alone leaves parcels systematically displaced relative to the anatomy
    they are named for.

    Returns (image, method) where method records which path was taken, so the
    coverage report always says whether a real warp happened.
    """
    if atlas_space == bold_space:
        return atlas_img, "same-space"

    try:
        from templateflow import api as tflow
        from nitransforms.io.itk import ITKCompositeH5
        from nitransforms.linear import Affine
        from nitransforms.manip import TransformChain
        from nitransforms.nonlinear import DenseFieldTransform
        from nitransforms.resampling import apply as nt_apply
    except ImportError:
        print(
            f"\nWARNING: atlas is in {atlas_space} but BOLD is in {bold_space}, and\n"
            f"  templateflow/nitransforms are not installed, so the atlas is being\n"
            f"  aligned by AFFINE ONLY. Parcels will be systematically displaced by\n"
            f"  a few mm relative to the anatomy they are named for.\n"
            f"  Fix: pip install templateflow nitransforms   (see external/atlases/README.md)\n"
            f"  Or set atlas.warp_to_bold_space: false in the config to silence this.\n",
            file=sys.stderr,
        )
        return atlas_img, "affine-only (WARNING: template mismatch)"

    xfm_path = tflow.get(
        bold_space, suffix="xfm", extension=".h5", **{"from": atlas_space}
    )
    if isinstance(xfm_path, list):
        xfm_path = xfm_path[0]
    reference = tflow.get(bold_space, resolution=2, suffix="T1w", desc=None, extension=".nii.gz")
    if isinstance(reference, list):
        reference = reference[0]

    # The TemplateFlow .h5 is an ITK composite: [affine, displacement field].
    # NOTE nitransforms.linear.load() also accepts this file but silently returns
    # ONLY the affine component, discarding the warp -- which would look like a
    # successful nonlinear warp while being affine-only. Build the chain explicitly.
    parts = ITKCompositeH5.from_filename(str(xfm_path))
    affine = Affine(parts[0].to_ras())
    field = DenseFieldTransform(parts[1], is_deltas=True)
    chain = TransformChain([field, affine])

    # order=0 (nearest neighbour): a label image must never be interpolated, or
    # non-existent labels appear at parcel boundaries.
    warped = nt_apply(chain, atlas_img, reference=str(reference), order=0)
    return warped, f"warped {atlas_space}->{bold_space}"


def load_atlas(cfg: dict) -> tuple[nib.Nifti1Image, np.ndarray, str]:
    """Load the parcellation, warp it into BOLD space, and list its labels."""
    atlas_path = resolve(cfg["atlas"]["image"])
    if not atlas_path.exists():
        raise FileNotFoundError(
            f"Atlas not found at {atlas_path}.\n"
            f"The {cfg['atlas']['name']} parcellation must be obtained separately "
            f"and placed there -- see external/atlases/README.md."
        )
    atlas_img = nib.load(atlas_path)

    method = "not attempted"
    if cfg["atlas"].get("warp_to_bold_space", False):
        atlas_img, method = warp_atlas_to_bold_space(
            atlas_img, cfg["atlas"].get("space", ""), cfg.get("bold_space", "")
        )

    labels = np.unique(np.asarray(atlas_img.dataobj).astype(np.int32))
    labels = labels[labels != 0]

    expected = cfg["atlas"].get("n_parcels")
    if expected and len(labels) != expected:
        # The base A424.nii.gz is missing label 120 (423 parcels); A424+2mm has
        # all 424. A count below expected usually means the wrong variant.
        print(
            f"WARNING: atlas has {len(labels)} non-zero labels, config expects {expected}. "
            f"Missing: {sorted(set(range(1, expected + 1)) - set(labels.tolist()))[:10]}",
            file=sys.stderr,
        )
    return atlas_img, labels, method


def _grid_key(img: nib.Nifti1Image) -> tuple:
    """Hashable identity of a voxel grid: shape plus affine."""
    return (tuple(img.shape[:3]), np.round(img.affine, 4).tobytes())


def get_parcel_indices(
    bold_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    labels: np.ndarray,
    cache: dict,
) -> list[np.ndarray]:
    """Voxel indices per parcel on this subject's grid, memoised by grid.

    A multi-site cohort typically has a few dozen distinct voxel grids at most,
    even across thousands of subjects, so resampling the atlas per subject would
    repeat the same computation many times over for each grid.
    """
    key = _grid_key(bold_img)
    if key not in cache:
        # Atlas -> subject grid. Nearest-neighbour: label images must not be
        # interpolated, or non-existent labels appear at parcel boundaries.
        atlas_on_bold = resample_to_img(
            atlas_img, bold_img, interpolation="nearest", force_resample=True, copy_header=True
        )
        flat_labels = np.asarray(atlas_on_bold.dataobj).astype(np.int32).reshape(-1)
        cache[key] = [np.flatnonzero(flat_labels == lab) for lab in labels]
    return cache[key]


def parcellate_subject(
    bold_path: Path, atlas_img: nib.Nifti1Image, labels: np.ndarray, cache: dict
) -> tuple[np.ndarray, int]:
    """Mean BOLD time series within each atlas parcel.

    Returns (n_parcels, n_timepoints) float32 and the number of empty parcels.
    """
    bold_img = nib.load(bold_path)
    parcel_idx = get_parcel_indices(bold_img, atlas_img, labels, cache)

    bold_data = np.asarray(bold_img.dataobj, dtype=np.float32)
    n_vol = bold_data.shape[3]
    flat = bold_data.reshape(-1, n_vol)

    out = np.full((len(labels), n_vol), np.nan, dtype=np.float32)
    n_empty = 0
    for i, idx in enumerate(parcel_idx):
        if idx.size == 0:
            n_empty += 1
            continue
        out[i] = flat[idx].mean(axis=0)
    return out, n_empty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--site", default=None, help="restrict to one site prefix, e.g. MI")
    parser.add_argument("--tr", type=float, default=None, help="restrict to one TR, e.g. 0.85")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.dataset)
    interim = resolve(cfg["interim_path"])

    manifest_path = interim / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run build_manifest.py first."
        )
    man = pd.read_csv(manifest_path)

    # Label-free exclusions only (unusable run lengths, unresolved TR mismatches).
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

    atlas_img, labels, warp_method = load_atlas(cfg)
    out_dir = interim / "parcels_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parcellating {len(man)} subjects with {cfg['atlas']['name']} ({len(labels)} parcels)")
    print(f"  atlas alignment: {warp_method}")
    coverage = []
    grid_cache: dict = {}
    for i, row in enumerate(man.itertuples(), 1):
        out_path = out_dir / f"{row.subject_id}.npy"
        if out_path.exists() and not args.overwrite:
            continue
        bold_path, _ = subject_files(cfg, row.subject_id)
        if bold_path is None:
            print(f"  SKIP {row.subject_id}: no BOLD file")
            continue
        try:
            ts, n_empty = parcellate_subject(bold_path, atlas_img, labels, grid_cache)
        except Exception as exc:
            print(f"  FAIL {row.subject_id}: {exc}")
            coverage.append({"subject_id": row.subject_id, "error": str(exc)})
            continue

        np.save(out_path, ts)
        coverage.append(
            {
                "subject_id": row.subject_id,
                "site": row.site,
                "n_parcels": ts.shape[0],
                "n_timepoints": ts.shape[1],
                "n_empty_parcels": n_empty,
                "n_nan_parcels": int(np.isnan(ts).all(axis=1).sum()),
                "atlas_alignment": warp_method,
            }
        )
        if i % 25 == 0 or i == len(man):
            print(f"  {i}/{len(man)}")

    if coverage:
        cov = pd.DataFrame(coverage)
        cov_path = interim / "parcellation_coverage.csv"
        cov.to_csv(cov_path, index=False)
        print(f"\nWrote coverage report -> {cov_path}")
        if "n_empty_parcels" in cov:
            bad = cov[cov["n_empty_parcels"] > 0]
            print(f"  subjects with empty parcels: {len(bad)}/{len(cov)}")
            if len(bad):
                print(f"  worst: {int(bad['n_empty_parcels'].max())} empty parcels")
    print(f"Parcel time series -> {out_dir}")


if __name__ == "__main__":
    main()
