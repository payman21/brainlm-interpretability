#!/usr/bin/env python3
"""Plan section 7.0 -- what does the representation encode about the signal?

Goodfire's first interpretability move: probe the frozen embeddings for properties
of the INPUT SIGNAL, not for the label. You cannot claim a model uses a feature it
does not encode.

Features are computed from the SAME 200-timepoint window the embedding came from,
so the probe is per-window: 22,482 samples rather than 938.

WHAT THE PROBE USES AS INPUT
----------------------------
The 512-d **token-mean** embedding -- the mean over all 4,240 parcel-time tokens of
the encoder's last hidden state (`mean` in frozen_embeddings.npz), not the CLS
token. The CLS token is probed alongside it as a second source, because BrainLM's
own attention analysis is built on CLS and the two need not agree.

FOUR EMBEDDING SOURCES, AND WHY
-------------------------------
A high probe score on its own says nothing. It has to be read against what the same
probe returns from representations that never learned anything:

    frozen_mean       pretrained BrainLM, token-mean            (the result)
    frozen_cls        pretrained BrainLM, CLS token             (does pooling matter)
    random_init_mean  randomly initialised BrainLM, token-mean  (does PRETRAINING matter)
    random_proj       fixed Gaussian projection of the flattened
                      424x200 window to 512-d                   (does the ARCHITECTURE matter)

A randomly initialised transformer over structured input is already a strong feature
extractor, so `random_init_mean` is the baseline that decides whether a high score is
about pretraining or about the shape of the data. `random_proj` is *linear* in the
input, so any quadratic feature (functional connectivity is a correlation) should be
out of its reach -- it separates "the encoder computed this" from "this survives any
random compression".

TARGETS THAT SHOULD SUCCEED, AND TARGETS THAT SHOULD FAIL
---------------------------------------------------------
A table of uniformly high scores means nothing without a row that comes out near
zero. Preprocessing gives us principled ones. The pipeline robust-scales each parcel
of each run by its own median and IQR, so every ABSOLUTE amplitude is divided out
before the model ever sees the signal. We compute those absolute quantities from
`interim/parcels_raw` -- the same parcels before scaling -- and probe for them too.
They are not recoverable in principle, and if they come back high the probe is
measuring something other than what we think.

    scale-invariant, the model can see these
      netfc_i_j     mean correlation within/between AA-7 networks i,j   (28)
      fc_mean       mean edge strength over all parcel pairs             (1)
      fc_sd         spread of edge strength                              (1)
      ac1_<net>     lag-1 autocorrelation, averaged per network          (7)
      slope_<net>   aperiodic spectral slope, per network                (7)
      falff_<net>   fractional low-frequency power, per network          (7)
      var_<net>     within-window SD of the SCALED signal, per network   (7)

    absolute amplitude, destroyed by robust scaling -- these should fail
      rawvar_<net>  within-window SD of the RAW signal, per network      (7)
      rawalff_<net> low-frequency amplitude of the RAW signal (0.01-0.08 Hz) (7)
      iqr_<net>     the per-parcel run IQR that scaling divided out      (7)

    identity of the input's provenance -- the window-level analogue of the
    reference study's locus and region-identity probes
      window_index  which of the run's disjoint windows this is (0-5)    (1)
      run_phase     phase-encoding direction of the run, LR vs RL        (1)
      session       which scanning session, REST1 vs REST2               (1)

`var_<net>` is the informative middle case: amplitude AFTER scaling is still visible
to the model, so it separates "absolute amplitude is gone" from "all amplitude is
gone". Parcel-level identity and temporal-patch position are probed at TOKEN level
by `probe_identity_tokens_hcp.py`; they cannot be asked of a window-mean embedding.

Probe: ridge from the 512-d embedding to each target, grouped 5-fold by subject so
no subject's windows straddle a split. Alpha is chosen per target by generalised
cross-validation inside the training fold. Reports out-of-fold R2.

Usage:
    python src/04_interpretability/probe_signal_features_hcp.py --n-subjects 938
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
from common import load_config, resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402

RUNS = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]
TR = 0.72
ALPHAS = np.logspace(-2, 6, 25)


def window_features(w: np.ndarray, net: np.ndarray, n_net: int, names: list[str]) -> dict:
    """Features of one 424 x 200 window of the SCALED signal.

    Unchanged from the original 51-feature version, plus `var_<net>`, so the
    scale-invariant numbers reproduce exactly. `net` is a 0-based network id per parcel.
    """
    x = w - w.mean(1, keepdims=True)
    sd = x.std(1)
    ok = sd > 0
    xz = np.zeros_like(x)
    xz[ok] = x[ok] / sd[ok, None]
    fc = (xz @ xz.T) / x.shape[1]
    np.fill_diagonal(fc, np.nan)

    out = {}
    iu = np.triu_indices(fc.shape[0], 1)
    edges = fc[iu]
    out["fc_mean"] = float(np.nanmean(edges))
    out["fc_sd"] = float(np.nanstd(edges))
    for i in range(n_net):
        for j in range(i, n_net):
            blk = fc[np.ix_(net == i, net == j)]
            out[f"netfc_{names[i]}__{names[j]}"] = float(np.nanmean(blk))

    # lag-1 autocorrelation per parcel
    ac1 = np.zeros(x.shape[0])
    ac1[ok] = (xz[ok, :-1] * xz[ok, 1:]).mean(1)

    # spectral slope and fALFF per parcel, from one periodogram
    f, pxx = sps.welch(x, fs=1.0 / TR, nperseg=min(128, x.shape[1]), axis=1)
    keep = f > 0
    lf = np.log10(f[keep])
    lp = np.log10(np.maximum(pxx[:, keep], 1e-20))
    lf_c = lf - lf.mean()
    slope = (lp - lp.mean(1, keepdims=True)) @ lf_c / (lf_c @ lf_c)
    band = (f >= 0.01) & (f <= 0.08)
    tot = pxx.sum(1)
    falff = np.divide(pxx[:, band].sum(1), tot, out=np.zeros_like(tot), where=tot > 0)

    for i in range(n_net):
        m = net == i
        out[f"ac1_{names[i]}"] = float(ac1[m].mean())
        out[f"slope_{names[i]}"] = float(slope[m].mean())
        out[f"falff_{names[i]}"] = float(falff[m].mean())
        out[f"var_{names[i]}"] = float(sd[m].mean())
    return out


def amplitude_features(wr: np.ndarray, iqr: np.ndarray, live: np.ndarray,
                       net: np.ndarray, n_net: int, names: list[str]) -> dict:
    """Absolute-amplitude features of one 424 x 200 window of the RAW, unscaled signal.

    These are what robust scaling removed. `iqr` is the per-parcel IQR over the whole
    run -- the divisor itself, constant across the run's windows and therefore the
    purest null: nothing about it reaches the model. `live` masks the one parcel the
    config zeroes (A424 label 405, a single voxel), which carries raw data the model
    never sees and would otherwise inflate a network mean.
    """
    x = wr - wr.mean(1, keepdims=True)
    sd = x.std(1)
    f, pxx = sps.welch(x, fs=1.0 / TR, nperseg=min(128, x.shape[1]), axis=1)
    band = (f >= 0.01) & (f <= 0.08)
    alff = np.sqrt(pxx[:, band].sum(1))

    out = {}
    for i in range(n_net):
        m = (net == i) & live
        out[f"rawvar_{names[i]}"] = float(sd[m].mean())
        out[f"rawalff_{names[i]}"] = float(alff[m].mean())
        out[f"iqr_{names[i]}"] = float(iqr[m].mean())
    return out


def family_of(name: str) -> str:
    if name.startswith("netfc"):
        return "netfc"
    return name.split("_")[0]


def oof_r2(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, folds: int) -> np.ndarray:
    """Out-of-fold R2 for every column of Y, from one ridge per fold.

    RidgeCV with alpha_per_target picks each target's alpha by generalised CV inside
    the training fold, so this is the same estimator as fitting each target
    separately -- it just shares the one decomposition instead of redoing it per
    target, which is what makes 79 targets x 4 sources tractable.
    """
    P = np.empty_like(Y)
    for tr, te in GroupKFold(folds).split(X, groups=groups):
        mx, sx = X[tr].mean(0), X[tr].std(0)
        sx[sx == 0] = 1.0
        my, sy = Y[tr].mean(0), Y[tr].std(0)
        sy[sy == 0] = 1.0
        m = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
        m.fit((X[tr] - mx) / sx, (Y[tr] - my) / sy)
        P[te] = m.predict((X[te] - mx) / sx) * sy + my
    sse = ((Y - P) ** 2).sum(0)
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1.0 - sse / sst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--n-subjects", type=int, default=938)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--proj-dim", type=int, default=512)
    ap.add_argument("--proj-seed", type=int, default=0)
    ap.add_argument("--sources", default="frozen_mean,frozen_cls,random_init_mean,random_proj")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    emb_dir = ROOT / f"results/{args.dataset}/brainlm/embeddings"
    sources = args.sources.split(",")

    def load_emb(fname: str, key: str) -> tuple[dict, np.ndarray]:
        z = np.load(emb_dir / fname, allow_pickle=True)
        idx = {(s, r, int(w)): i for i, (s, r, w) in
               enumerate(zip(z["subject_id"].astype(str), z["run"].astype(str), z["window"]))}
        return idx, np.asarray(z[key])       # decompress ONCE; z[key] re-inflates on access

    banks = {}
    if "frozen_mean" in sources:
        banks["frozen_mean"] = load_emb("frozen_embeddings.npz", "mean")
    if "frozen_cls" in sources:
        banks["frozen_cls"] = load_emb("frozen_embeddings.npz", "cls")
    if "random_init_mean" in sources:
        p = emb_dir / "random_init_embeddings.npz"
        if not p.exists():
            raise SystemExit(
                f"{p} missing. Produce it with:\n"
                f"  python src/02_finetuning/extract_embeddings.py --dataset {args.dataset} "
                f"--model brainlm --device mps --windows all --random-init")
        banks["random_init_mean"] = load_emb("random_init_embeddings.npz", "mean")

    lab = pd.read_csv(resolve("external/atlases/A424/A424_Labels_AA-AAc_main_maps.csv"))
    lab.columns = [c.strip().lstrip("﻿") for c in lab.columns]
    net_name = lab.iloc[:, 5].values
    names = list(pd.unique(net_name))
    short = {n: n.split()[0][:4] + ("" if len(n.split()) == 1 else n.split()[1][:3])
             for n in names}
    net = np.array([names.index(n) for n in net_name])
    n_net = len(names)
    sn = [short[n] for n in names]
    print(f"{n_net} networks: {sn}")

    cohort = [l.strip() for l in open(resolve(cfg["cohort"]["subject_list"])) if l.strip()]
    ts_dir = resolve(cfg["processed_path"]) / "timeseries"
    raw_dir = resolve(cfg["interim_path"]) / "parcels_raw"
    subs = [s for s in cohort if (ts_dir / f"{s}_{RUNS[0]}.npy").exists()][: args.n_subjects]

    rng = np.random.default_rng(args.proj_seed)
    proj = None                                   # built lazily, once the window shape is known

    rows, groups, keys, projs = [], [], [], []
    for i, s in enumerate(subs, 1):
        for r in RUNS:
            p, pr = ts_dir / f"{s}_{r}.npy", raw_dir / f"{s}_{r}.npy"
            if not p.exists() or not pr.exists():
                continue
            a = np.load(p).astype(np.float32)
            ar = np.load(pr).astype(np.float32)
            live = ar.std(1) > 0
            q = np.percentile(ar, [25, 75], axis=1)
            iqr = (q[1] - q[0]).astype(np.float64)
            n_win = a.shape[1] // 200
            for w in range(n_win):
                seg, segr = a[:, w * 200:(w + 1) * 200], ar[:, w * 200:(w + 1) * 200]
                if seg.shape[1] < 200:
                    continue
                d = window_features(seg, net, n_net, sn)
                d.update(amplitude_features(segr, iqr, live, net, n_net, sn))
                d["window_index"] = float(w)
                d["runphase_LRvsRL"] = float(r.endswith("_RL"))
                d["session_1vs2"] = float("REST2" in r)
                rows.append(d)
                groups.append(s)
                keys.append((s, r, w))
                if "random_proj" in sources:
                    flat = seg.ravel().astype(np.float32)
                    if proj is None:
                        proj = rng.standard_normal(
                            (flat.size, args.proj_dim)).astype(np.float32) / np.sqrt(flat.size)
                    projs.append(flat @ proj)
        if i % 50 == 0 or i == len(subs):
            print(f"  {i}/{len(subs)} subjects, {len(rows)} windows", flush=True)

    F = pd.DataFrame(rows)
    g = np.array(groups)
    print(f"\n{len(F)} windows x {F.shape[1]} targets")

    if "random_proj" in sources:
        banks["random_proj"] = ({k: i for i, k in enumerate(keys)}, np.stack(projs))

    # Only targets that are finite and non-constant everywhere are probeable.
    cols = [c for c in F.columns
            if np.isfinite(F[c].values).all() and F[c].values.std() > 0]
    Y = F[cols].values.astype(np.float64)

    res = []
    for src in sources:
        idx, bank = banks[src]
        sel = np.array([idx.get(k, -1) for k in keys])
        m = sel >= 0
        if not m.all():
            print(f"  {src}: {(~m).sum()} of {len(keys)} windows have no embedding, dropped")
        X = bank[sel[m]].astype(np.float64)
        r2 = oof_r2(X, Y[m], g[m], args.folds)
        print(f"{src}: {X.shape[0]} windows x {X.shape[1]}-d")
        res += [{"source": src, "feature": c, "family": family_of(c), "r2": float(v),
                 "n_windows": int(m.sum())} for c, v in zip(cols, r2)]

    df = pd.DataFrame(res)
    piv = df.pivot_table(index="family", columns="source", values="r2", aggfunc="mean")
    piv = piv.reindex(columns=[s for s in sources if s in piv.columns])
    order = ["fc", "ac1", "slope", "falff", "netfc", "var", "rawvar", "rawalff", "iqr",
             "window", "runphase", "session"]
    piv = piv.reindex([o for o in order if o in piv.index])
    print("\n=== mean out-of-fold R2 by family and embedding source ===")
    print(piv.to_string(float_format=lambda v: f"{v:7.3f}"))

    out = ROOT / "results" / args.dataset / "brainlm" / "interpretability"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "probe_signal_features.csv", index=False)
    F.assign(subject=g).to_csv(out / "signal_features_per_window.csv.gz",
                               index=False, compression="gzip")
    best = df[df.source == "frozen_mean"].sort_values("r2", ascending=False).iloc[0]
    stamp(out, "src/04_interpretability/probe_signal_features_hcp.py", ROOT,
          n_windows=int(len(F)), n_subjects=len(subs), n_targets=len(cols),
          sources=sources, best_feature=str(best.feature), best_r2=float(best.r2))
    print(f"\n-> {out}/probe_signal_features.csv")


if __name__ == "__main__":
    main()
