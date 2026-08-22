"""Parcellate one HCP rfMRI run to A424 parcel means, with verification.

Deletion of the source NIfTI is gated on this script exiting 0. Every check
that could distinguish a good parcellation from a silently corrupt one runs
here, including a read-back of the written .npy, so a zero exit means the
parcel file on disk is known-good and the voxels are safe to discard.

Prints one JSON record to stdout. Exit 0 = verified, 1 = failed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

N_PARCELS = 424
EXPECT_SHAPE = (91, 109, 91)
FULL_T = 1200
# 15 of the 1016 subjects have runs the scanner cut short (907-1177 TP); the
# phenotype column 3T_RS-fMRI_PctCompl flags exactly these. A short run is
# still usable -- BrainLM consumes disjoint 200-TP windows, so 907 TP yields 4
# instead of 6 -- so accept anything with at least two full windows.
MIN_T = 400
MAX_DEGENERATE_PARCELS = 10
EXPECT_AFFINE = np.array([[-2., 0., 0., 90.],
                          [0., 2., 0., -126.],
                          [0., 0., 2., -72.],
                          [0., 0., 0., 1.]])


def fail(msg: str, **extra):
    print(json.dumps({"ok": False, "error": msg, **extra}))
    sys.exit(1)


def main():
    nii_path, atlas_path, out_path, expect_bytes = sys.argv[1:5]
    nii_path, out_path = Path(nii_path), Path(out_path)
    expect_bytes = int(expect_bytes)

    # ---- 1. the bytes we fetched are the bytes S3 has -------------------
    actual = nii_path.stat().st_size
    if actual != expect_bytes:
        fail("size mismatch", expected=expect_bytes, actual=actual)

    # ---- 2. the NIfTI is readable and on the grid A424 assumes ----------
    try:
        img = nib.load(str(nii_path))
    except Exception as e:  # truncated or corrupt gzip lands here
        fail(f"nifti load failed: {e}")
    if tuple(img.shape[:3]) != EXPECT_SHAPE:
        fail("unexpected spatial shape", shape=list(img.shape))
    if img.ndim != 4 or img.shape[3] < MIN_T:
        fail("too few timepoints", shape=list(img.shape), min_t=MIN_T)
    n_t = int(img.shape[3])
    if not np.allclose(img.affine, EXPECT_AFFINE, atol=1e-4):
        fail("affine does not match A424 grid", affine=img.affine.tolist())

    atlas = np.asarray(nib.load(atlas_path).dataobj).astype(np.int32)
    if atlas.shape != EXPECT_SHAPE:
        fail("atlas shape mismatch", shape=list(atlas.shape))
    flat = atlas.ravel()
    mask = flat > 0
    idx = flat[mask]
    counts = np.bincount(idx, minlength=N_PARCELS + 1)[1:]
    if int((counts > 0).sum()) != N_PARCELS:
        fail("atlas does not cover 424 parcels", covered=int((counts > 0).sum()))

    # ---- 3. parcellate: mean over voxels, per parcel, per timepoint -----
    # Reading the whole 4.3 GB volume at once would blow the per-core memory
    # budget, so stream it in blocks of timepoints.
    out = np.empty((N_PARCELS, n_t), np.float32)
    denom = np.maximum(counts, 1)[:, None]
    try:
        for s in range(0, n_t, 200):
            e = min(s + 200, n_t)
            blk = np.asarray(img.dataobj[..., s:e], dtype=np.float32).reshape(-1, e - s)[mask]
            for j in range(e - s):
                out[:, s + j] = np.bincount(idx, weights=blk[:, j], minlength=N_PARCELS + 1)[1:]
    except Exception as ex:  # a truncated gzip usually surfaces mid-read
        fail(f"read/parcellate failed: {ex}")
    out /= denom

    # ---- 4. the parcels are usable, not silently degenerate -------------
    if out.shape != (N_PARCELS, n_t):
        fail("bad output shape", shape=list(out.shape))
    if out.dtype != np.float32:
        fail("bad dtype", dtype=str(out.dtype))
    if not np.isfinite(out).all():
        fail("non-finite values", n_nan=int(np.isnan(out).sum()), n_inf=int(np.isinf(out).sum()))
    # A handful of parcels legitimately carry no signal: frontopolar and
    # orbitofrontal parcels sit over the sinuses and drop out in some runs
    # (e.g. 270 = L_10pp). BrainLM's toolkit already treats zero as missing,
    # and the pipeline zeroes parcel 405 by convention, so a small count is
    # data, not corruption. A large count means something is actually wrong.
    flat = np.where(out.std(axis=1) == 0)[0] + 1
    zero = np.where((out == 0).all(axis=1))[0] + 1
    if len(flat) > MAX_DEGENERATE_PARCELS:
        fail("too many degenerate parcels", flat_parcels=flat.tolist())
    gmean = float(out.mean())
    if not (1.0 < gmean < 1e6):
        fail("implausible BOLD magnitude", grand_mean=gmean)

    # ---- 5. write, then read back and compare ---------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save appends ".npy" to any path lacking it, so write through an open
    # handle instead of letting it rewrite the temp filename underneath us.
    tmp = out_path.parent / (out_path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, out)
    reread = np.load(tmp)
    if reread.shape != out.shape or reread.dtype != out.dtype or not np.array_equal(reread, out):
        tmp.unlink(missing_ok=True)
        fail("read-back mismatch -- refusing to certify")
    tmp.replace(out_path)

    print(json.dumps({
        "ok": True,
        "out": str(out_path),
        "src_bytes": actual,
        "out_bytes": out_path.stat().st_size,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "timepoints": n_t,
        "n_windows": n_t // 200,
        "short_run": n_t < FULL_T,
        "grand_mean": round(gmean, 2),
        "zero_parcels": zero.tolist(),
        "flat_parcels": flat.tolist(),
        "parcel_sd_min": round(float(out.std(axis=1).min()), 4),
    }))


if __name__ == "__main__":
    main()
