#!/usr/bin/env python3
"""Extract frozen BrainLM embeddings for every subject with a model-ready timeseries.

Generic: takes --dataset and --model and reads both configs. No dataset constants.

Two embeddings are written per subject, both 512-dimensional:

  cls   the CLS token (index 0 of the encoder output). This is the representation
        BrainLM's own attention analysis is built on.
  mean  the mean over all 4,240 parcel-time tokens.

Masking is disabled (``embedding.mask_ratio: 0``) so the encoder sees every token
and the result is deterministic. The checkpoint's own mask_ratio is a pretraining
setting; leaving it on would make each embedding depend on a random draw.

The input gain and coordinate convention come from the model config, where they
were frozen after a label-free reconstruction sweep. They are not arguments here,
so they cannot drift between runs.

Usage:
    python src/02_finetuning/extract_embeddings.py --dataset hcp_ya --model brainlm
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
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
from common import load_config, resolve  # noqa: E402


def load_model_config(name: str) -> dict:
    with (ROOT / "configs" / "models" / f"{name}.yaml").open() as fh:
        return yaml.safe_load(fh)


def identity_noise(model, device: str):
    """Monotonic noise, so BrainLM's argsort-based shuffle becomes the identity.

    REQUIRED for reproducibility. The encoder is built from NystromformerLayer,
    whose attention approximation computes landmarks as segment means over
    CONSECUTIVE tokens, so its output depends on token ORDER. BrainLM's
    random_masking() draws torch.rand and argsorts it on every call, permuting all
    4,240 tokens even at mask_ratio 0 where it drops none. Left alone, the same
    subject and window give a different embedding on every call -- measured at 3.0%
    median relative spread on the CLS token, on a single device.

    Passing monotonic noise makes argsort the identity, preserving the parcel-major
    order the tokens were built in, and makes the output bit-identical across calls.
    """
    n = model.config.num_brain_voxels * (
        model.config.num_timepoints_per_voxel // model.config.timepoint_patching_size
    )
    return (torch.arange(n, dtype=torch.float32, device=device) / n).unsqueeze(0)


def load_coords(path: Path, n_parcels: int, units: str) -> torch.Tensor:
    arr = np.loadtxt(path)
    if arr.shape != (n_parcels, 4):
        raise ValueError(f"expected ({n_parcels}, 4) in {path}, got {arr.shape}")
    if not (arr[:, 0] == np.arange(1, n_parcels + 1)).all():
        raise ValueError(f"{path} is not in ascending label order; parcel mapping would be wrong")
    xyz = arr[:, 1:].astype(np.float32)
    if units == "unit":
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        xyz = (xyz - lo) / (hi - lo)
    elif units != "mm":
        raise ValueError(f"unknown coordinate_units {units!r}")
    return torch.from_numpy(xyz)


def centre_window(ts: np.ndarray, n: int) -> np.ndarray:
    start = (ts.shape[1] - n) // 2
    return ts[:, start : start + n]


def disjoint_windows(ts: np.ndarray, n: int) -> list[np.ndarray]:
    """Every complete non-overlapping window, front-aligned.

    A 1200-timepoint HCP run gives 6; a short run gives fewer (the cohort's
    shortest, 907 TP, gives 4). The remainder at the end is dropped rather than
    padded -- a partial window is not something the model was pretrained on.
    """
    return [ts[:, i : i + n] for i in range(0, ts.shape[1] - n + 1, n)]


def split_run_id(stem: str) -> tuple[str, str]:
    """``100206_rfMRI_REST1_LR`` -> ``("100206", "rfMRI_REST1_LR")``.

    Datasets with one file per subject and no run in the name return the stem
    unchanged with an empty run.
    """
    subject, sep, run = stem.partition("_")
    return (subject, run) if sep else (stem, "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--model", default="brainlm")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--random-init",
        action="store_true",
        help="randomly initialised weights instead of the checkpoint. THE control for whether "
        "pretraining contributes anything: randomly initialised transformers are strong feature "
        "extractors, so any probe score must be read against this, not against chance.",
    )
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument(
        "--windows",
        choices=["centre", "all"],
        default="centre",
        help="'centre' = one middle window per file (one embedding per file); "
             "'all' = every disjoint window, one embedding each. HCP wants 'all': "
             "its 4 runs x 6 windows are 24 embeddings per subject, and taking only "
             "the centre one would discard 23/24 of the cohort.",
    )
    args = ap.parse_args()

    cfg, mcfg = load_config(args.dataset), load_model_config(args.model)
    sys.path.insert(0, str(resolve(mcfg["code_path"])))
    from brainlm_mae.modeling_brainlm import BrainLMForPretraining

    if args.random_init:
        from brainlm_mae.configuration_brainlm import BrainLMConfig

        torch.manual_seed(args.init_seed)
        model = BrainLMForPretraining(BrainLMConfig.from_pretrained(resolve(mcfg["checkpoint_path"])))
    else:
        model = BrainLMForPretraining.from_pretrained(resolve(mcfg["checkpoint_path"]))
    # Disable masking on both the model config and the embeddings module, because
    # random_masking reads it from whichever object it was handed.
    ratio = float(mcfg["embedding"]["mask_ratio"])
    model.config.mask_ratio = ratio
    model.vit.embeddings.config.mask_ratio = ratio
    model.eval().to(args.device)

    noise = identity_noise(model, args.device)
    n_parcels = model.config.num_brain_voxels
    n_time = int(mcfg["input"]["window"])
    gain = float(mcfg["input"]["gain"])
    xyz = load_coords(
        resolve(mcfg["input"]["coordinates"]), n_parcels, mcfg["input"]["coordinate_units"]
    ).unsqueeze(0).to(args.device)
    print(f"{args.model}: mask_ratio={ratio}, gain={gain}, window={n_time}, "
          f"device={args.device}, token_order={mcfg['embedding']['token_order']}")

    processed = resolve(cfg["processed_path"])
    subjects = sorted(p.stem for p in (processed / "timeseries").glob("*.npy"))
    if args.limit:
        subjects = subjects[: args.limit]

    cls_rows, mean_rows, kept, runs, widx, skipped = [], [], [], [], [], []
    for i, sid in enumerate(subjects, 1):
        ts = np.load(processed / "timeseries" / f"{sid}.npy")
        if ts.shape[1] < n_time or not np.isfinite(ts).all():
            skipped.append({"subject_id": sid, "reason": f"{ts.shape[1]} frames or non-finite"})
            continue
        subject, run = split_run_id(sid)
        wins = (
            disjoint_windows(ts, n_time) if args.windows == "all"
            else [centre_window(ts, n_time)]
        )
        for w, win in enumerate(wins):
            sig = torch.from_numpy(np.ascontiguousarray(win)).float().unsqueeze(0).to(args.device)
            with torch.no_grad():
                out = model.vit(
                    signal_vectors=sig * gain, xyz_vectors=xyz, noise=noise, return_dict=True
                )
            h = out.last_hidden_state[0]       # [1 + n_tokens, hidden]
            cls_rows.append(h[0].cpu().numpy())
            mean_rows.append(h[1:].mean(0).cpu().numpy())
            kept.append(subject)
            runs.append(run)
            widx.append(w)
        if i % 100 == 0 or i == len(subjects):
            print(f"  {i}/{len(subjects)} files, {len(cls_rows)} embeddings", flush=True)

    # Canonical results root, keyed by the dataset name as given on the command
    # line. Case matters: macOS merges differing-case directories silently, Linux
    # would not, so this is pinned rather than derived from cfg["name"].
    dataset_dir = args.dataset
    out_dir = ROOT / "results" / dataset_dir / args.model / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / ("random_init_embeddings.npz" if args.random_init else "frozen_embeddings.npz"),
        subject_id=np.array(kept),
        run=np.array(runs),
        window=np.array(widx, dtype=np.int16),
        cls=np.stack(cls_rows).astype(np.float32),
        mean=np.stack(mean_rows).astype(np.float32),
    )
    tag = "random_init" if args.random_init else "frozen"
    print(f"\n{len(kept)} subjects -> {out_dir / (tag + '_embeddings.npz')}")
    if skipped:
        pd.DataFrame(skipped).to_csv(out_dir / "embedding_skipped.csv", index=False)
        print(f"{len(skipped)} skipped -> {out_dir / 'embedding_skipped.csv'}")


if __name__ == "__main__":
    main()
