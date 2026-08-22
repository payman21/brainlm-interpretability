#!/usr/bin/env python3
"""Plan 7.4 -- BatchTopK sparse autoencoder on BrainLM token activations.

Goodfire's unsupervised step. A sparse overcomplete dictionary decomposes the
encoder's last-layer activations into features, each of which can then be
characterised by what makes it fire.

BatchTopK (Bussmann et al. 2024): keep the k*batch largest pre-activations across
the WHOLE batch rather than the top-k per example, so the sparsity budget is shared
and features are not forced into every sample. Same objective as a TopK SAE, better
behaved dictionaries.

ATTRIBUTION: see sae_attribution_hcp.py, which decomposes the model's OWN prediction
over these features in closed form. An earlier version of this pipeline replaced that
step with a sparse linear model fitted from feature activations to the behavioural
score. That answers a different question -- what predicts the outcome, rather than what
the model relies on -- and has been dropped. The closed form needs no gradients, so the
reason the original detour existed (gradients through this encoder are
architecture-dominated; see docs/hcp_ya_findings_log.md 5.2) no longer applies.

TWO THINGS THIS SCRIPT MUST DO for that attribution to be sound:

  --subjects-file   Restrict training to the training split. An SAE fitted on all 938
                    subjects has seen the holdout, and every attribution built on its
                    dictionary inherits that contact. Unsupervised is not the same as
                    uninvolved.
  frozen threshold  BatchTopK's cut is a property of whatever batch encode() was
                    handed, so an SAE re-run on new data in different batches does not
                    reproduce its own training-time activations. The cut taken over the
                    whole training matrix is computed once and stored in sae.pt as
                    `thresh`; inference uses encode_fixed() with that constant, which
                    makes the encoder a JumpReLU with a fixed cut.

Step 2 -- characterising features against parcel, network, temporal patch and signal
statistics -- lives in characterise_sae_hcp.py and is where the interpretation happens.

Usage:
    python src/04_interpretability/train_sae_hcp.py --width 4096 --k 32 --epochs 30 \
        --subjects-file results/hcp_ya/splits/train_subjects.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common_provenance import stamp  # noqa: E402


class BatchTopKSAE(nn.Module):
    def __init__(self, d_in: int, width: int, k: int):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d_in, width)
        self.dec = nn.Linear(width, d_in, bias=False)
        # Tied init, unit-norm decoder columns: standard, keeps feature scale
        # from drifting into the decoder where it stops being comparable.
        with torch.no_grad():
            w = torch.randn(width, d_in) / np.sqrt(d_in)
            self.dec.weight.copy_(w.T)
            self.enc.weight.copy_(w)
            self.enc.bias.zero_()
            self.normalise_decoder()
        self.b_pre = nn.Parameter(torch.zeros(d_in))

    @torch.no_grad()
    def normalise_decoder(self):
        self.dec.weight /= self.dec.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def encode(self, x):
        z = torch.relu(self.enc(x - self.b_pre))
        # BatchTopK: one shared budget over the whole batch.
        n_keep = max(1, self.k * z.shape[0])
        flat = z.flatten()
        if n_keep < flat.numel():
            thresh = torch.topk(flat, n_keep, sorted=False).values.min()
            z = z * (z >= thresh)
        return z

    def encode_fixed(self, x, thresh: float):
        """encode() with the batch-derived cut replaced by a constant.

        This is what inference must use. encode() would otherwise re-derive the cut from
        whatever rows happen to be in the call, so the same window would get different
        activations depending on how it was batched.
        """
        z = torch.relu(self.enc(x - self.b_pre))
        return z * (z >= thresh)

    @torch.no_grad()
    def global_threshold(self, X: torch.Tensor, chunk: int = 8192) -> float:
        """The BatchTopK cut taken over the WHOLE matrix X, exactly.

        Chunked because the full pre-activation matrix is len(X) x width -- at 500k
        tokens and width 4096 that is 8 GB in float32, and materialising it to call
        topk once is how this gets killed on a smaller machine. A running buffer of the
        top n_keep values gives the same answer in ~50 MB.
        """
        n_keep = max(1, self.k * X.shape[0])
        buf = None
        for i in range(0, X.shape[0], chunk):
            # The running buffer and its topk stay on CPU: n_keep here is in the tens of
            # millions, and topk at that size is where MPS and small GPUs fall over.
            # Only the matmul needs the accelerator.
            z = torch.relu(self.enc(X[i:i + chunk] - self.b_pre)).flatten().cpu()
            buf = z if buf is None else torch.cat([buf, z])
            if buf.numel() > n_keep:
                buf = torch.topk(buf, n_keep, sorted=False).values
        # Fewer values than the budget means encode() would not have thresholded at
        # all; 0.0 reproduces that (relu already clamps the negatives).
        return float(buf.min()) if buf.numel() >= n_keep else 0.0

    def forward(self, x):
        z = self.encode(x)
        return self.dec(z) + self.b_pre, z


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--model", default="brainlm")
    ap.add_argument("--acts", type=Path, default=None)
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--k", type=int, default=32, help="avg active features per token")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--subjects-file", type=Path, default=None,
                    help="restrict training activations to these subject ids, one per "
                         "line. Use results/<dataset>/splits/train_subjects.txt -- "
                         "without it the dictionary is fitted on the holdout too")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where sae.pt and sae_features.npz go. Default is the canonical "
                         "results/<dataset>/<model>/interpretability/sae; point it "
                         "elsewhere for a smoke test so a short run cannot be mistaken "
                         "for the real dictionary")
    args = ap.parse_args()

    interp = ROOT / "results" / args.dataset / args.model / "interpretability"
    z = np.load(args.acts or (interp / "token_activations.npz"), allow_pickle=True)
    A = z["acts"].astype(np.float32)
    meta = np.asarray(z["meta"])
    torch.manual_seed(args.seed)

    # Filter by subject BEFORE anything else, including the mu/scale normalisation
    # constants, which are themselves fitted quantities. `window_row` indexes the
    # per-window arrays (subject/run/window), so those stay unfiltered while the
    # per-activation arrays are subset.
    if args.subjects_file:
        keep_subj = {l.strip() for l in open(args.subjects_file) if l.strip()}
        wrow = meta[:, list(z["meta_cols"]).index("window_row")].astype(int)
        subj_of_act = np.asarray(z["subject"]).astype(str)[wrow]
        m = np.isin(subj_of_act, list(keep_subj))
        if not m.any():
            raise SystemExit(f"no activations match the subjects in {args.subjects_file}")
        print(f"subject filter: {m.sum():,}/{len(m):,} activations from "
              f"{len(set(subj_of_act[m]))} of {len(set(subj_of_act))} sampled subjects")
        A, meta = A[m], meta[m]

    # Centre and scale to unit mean-norm: SAE reconstruction loss is scale
    # dependent, and this makes the loss comparable across runs.
    mu = A.mean(0, keepdims=True)
    Ac = A - mu
    scale = float(np.linalg.norm(Ac, axis=1).mean())
    Ac /= scale
    X = torch.from_numpy(Ac)
    n_val = int(len(X) * args.val_frac)
    perm = torch.randperm(len(X))
    Xv, Xt = X[perm[:n_val]].to(args.device), X[perm[n_val:]].to(args.device)
    print(f"activations {tuple(A.shape)} | train {len(Xt):,} val {len(Xv):,} | "
          f"width {args.width} k {args.k} | device {args.device}")

    sae = BatchTopKSAE(A.shape[1], args.width, args.k).to(args.device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)

    def fvu(x, xh):
        return float(((x - xh) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        sae.train()
        idx = torch.randperm(len(Xt), device=args.device)
        tot = 0.0
        for i in range(0, len(Xt), args.batch_size):
            xb = Xt[idx[i:i + args.batch_size]]
            xh, _ = sae(xb)
            loss = ((xb - xh) ** 2).sum(-1).mean()
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            sae.normalise_decoder()
            tot += float(loss.detach())
        sae.eval()
        with torch.no_grad():
            xh, zv = sae(Xv)
            v = fvu(Xv, xh)
            alive = int((zv > 0).any(0).sum())
            l0 = float((zv > 0).float().sum(1).mean())
        print(f"  ep{ep:>3} train {tot/max(1,len(Xt)//args.batch_size):.4f} | "
              f"val FVU {v:.4f} | L0 {l0:.1f} | alive {alive}/{args.width} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # Freeze the BatchTopK cut over the whole training matrix, then encode with THAT
    # constant rather than with encode()'s per-batch cut. Chunked so the width-4096
    # pre-activation matrix is never materialised in full.
    Xd = X.to(args.device)
    thresh = sae.global_threshold(Xd)
    with torch.no_grad():
        Z = np.empty((len(Xd), args.width), dtype=np.float32)
        for i in range(0, len(Xd), args.batch_size):
            Z[i:i + args.batch_size] = (
                sae.encode_fixed(Xd[i:i + args.batch_size], thresh).cpu().numpy())
    alive = (Z > 0).any(0)
    l0_fixed = float((Z > 0).sum(1).mean())
    print(f"\nfinal: val FVU {v:.4f}, L0 {l0:.1f}, {int(alive.sum())}/{args.width} alive features")
    print(f"frozen threshold {thresh:.6g} -> L0 {l0_fixed:.1f} over the full matrix "
          f"(target k={args.k}; these agree by construction, a large gap means a bug)")

    out = args.out_dir or (interp / "sae")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sae.state_dict(), "width": args.width, "k": args.k,
                "mu": mu, "scale": scale, "thresh": thresh}, out / "sae.pt")
    np.savez_compressed(out / "sae_features.npz", Z=Z, alive=alive,
                        meta=meta, meta_cols=z["meta_cols"],
                        subject=z["subject"], run=z["run"], window=z["window"])
    stamp(out, "src/04_interpretability/train_sae_hcp.py", ROOT,
          width=args.width, k=args.k, epochs=args.epochs,
          val_fvu=float(v), l0=float(l0), l0_fixed_threshold=l0_fixed,
          thresh=float(thresh), alive=int(alive.sum()),
          n_activations=int(len(A)),
          subjects_file=str(args.subjects_file) if args.subjects_file else None)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
