#!/usr/bin/env python3
"""Shared helpers for the preprocessing scripts.

Kept deliberately small: config loading, subject discovery, and the confound
handling that ``build_manifest.py`` and ``prepare_timeseries.py`` both need.
Nothing dataset-specific is hard-coded here -- values come from
``configs/datasets/<dataset>.yaml``.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(dataset: str) -> dict:
    """Read ``configs/datasets/<dataset>.yaml``."""
    path = REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No dataset config at {path}")
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve(path_str: str) -> Path:
    """Resolve a config path: absolute paths as-is, relative ones off the repo root."""
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def site_of(subject_id: str) -> str:
    """Site prefix of a subject ID (``sub-MI00452`` -> ``MI``).

    NOTE: site is *inferred* from the ID prefix. The prefix -> centre -> country
    mapping is not documented anywhere in the local copy of the dataset; it is an
    outstanding question for the data provider.
    """
    return "".join(c for c in subject_id.removeprefix("sub-") if c.isalpha())


def find_subjects(cfg: dict) -> list[str]:
    """Subject directories present on the raw drive, sorted."""
    raw = resolve(cfg["raw_path"])
    if not raw.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw}. Is the external drive mounted?"
        )
    return sorted(
        d.name for d in raw.iterdir() if d.is_dir() and d.name.startswith("sub-")
    )


def subject_files(cfg: dict, subject_id: str) -> tuple[Path | None, Path | None]:
    """Locate the BOLD and confounds files for one subject.

    Globs rather than string-formats because filename entities are inconsistent
    across sites (``task-resting`` / ``task-rest`` / ``task-RESTING``, with and
    without ``ses-``, ``dir-``, ``run-``). See the attributes doc, section 2.2.
    """
    subj_dir = resolve(cfg["raw_path"]) / subject_id
    bold = sorted(glob.glob(str(subj_dir / cfg["cohort"]["bold_glob"])))
    conf = sorted(glob.glob(str(subj_dir / cfg["cohort"]["confounds_glob"])))
    return (Path(bold[0]) if bold else None, Path(conf[0]) if conf else None)


def read_bold_sidecar(bold_path: Path) -> dict:
    """Read the JSON sidecar sitting next to a preprocessed BOLD file."""
    sidecar = Path(str(bold_path).replace(".nii.gz", ".json"))
    if not sidecar.exists():
        return {}
    with open(sidecar) as fh:
        return json.load(fh)


def load_confounds(confounds_path: Path) -> pd.DataFrame:
    """Load an fMRIPrep confounds TSV.

    fMRIPrep writes ``n/a`` in the first row of every derivative column and in
    the first value of ``framewise_displacement`` / ``dvars``. Those become NaN
    and are zero-filled -- leaving them would propagate NaN through the whole
    design matrix on the first timepoint.
    """
    df = pd.read_csv(confounds_path, sep="\t", na_values=["n/a"])
    return df.fillna(0.0)


def expand_confounds(
    df: pd.DataFrame, base_columns: list[str], expansions: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Select the requested expansions of a set of base confound columns.

    fMRIPrep already ships ``x``, ``x_derivative1``, ``x_power2`` and
    ``x_derivative1_power2``, so this selects rather than recomputes them.
    Missing columns raise: silently dropping a regressor would make the
    denoising strategy vary per subject, which is precisely what the
    fixed-width design exists to prevent.
    """
    suffixes = {
        "raw": "",
        "derivative1": "_derivative1",
        "power2": "_power2",
        "derivative1_power2": "_derivative1_power2",
    }
    names: list[str] = []
    for base in base_columns:
        for exp in expansions:
            if exp not in suffixes:
                raise ValueError(f"Unknown expansion {exp!r}")
            names.append(f"{base}{suffixes[exp]}")

    missing = [n for n in names if n not in df.columns]
    if missing:
        raise KeyError(f"Confound columns absent: {missing}")
    return df[names].to_numpy(dtype=np.float64), names


def censor_mask(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """Boolean mask of frames flagged by fMRIPrep; True = flagged.

    Each ``motion_outlier_XX`` column is a one-hot indicator for a single frame
    exceeding FD > 0.5 mm or std_dvars > 1.5.
    """
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return np.zeros(len(df), dtype=bool)
    return df[cols].to_numpy().sum(axis=1) > 0


def count_prefix(df: pd.DataFrame, prefix: str) -> int:
    """Number of confound columns whose name starts with ``prefix``."""
    return sum(1 for c in df.columns if c.startswith(prefix))
