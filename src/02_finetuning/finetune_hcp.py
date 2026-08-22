#!/usr/bin/env python3
"""Fine-tune BrainLM on HCP-YA for a continuous subject-level target.

Plan section 4.2. Reached when frozen embeddings (4.1) predict but not well
enough: on PMAT24_A_CR they give out-of-fold R2 0.075 / r 0.284, which is the
number to beat.

LADDER
------
Two rungs, cheapest first, because with 938 subjects against 13.4M parameters
nobody can say in advance where the capacity sweet spot is:

    last_block   head + the top encoder block   (2.50M trainable)
    all          everything                     (8.82M trainable)

The `head` rung is deliberately absent: it is the frozen setting already measured
in 4.1, where an MLP head scored *worse* than ridge (R2 0.026 vs 0.075). Training
the same head by SGD re-learns a known answer.

RECIPE
------
Follows BrainLM section 3.4 where it specifies something: 3-layer MLP on the CLS
token, 40% dropout on encoder activations and head, target min-max scaled to
[0, 1]. Differential learning rates because one rate either destroys pretrained
features or leaves the head undertrained.

Windows: ALL disjoint 200-timepoint windows are used every epoch (24 per subject
for most), shuffled across subjects so a batch mixes them rather than processing
one subject's windows together. Training is per-window; evaluation predicts each
window and averages within subject, the aggregation that won in 4.1.

Folds are subject-level. NOTE this does not control family structure -- HCP-YA is
a twin study and Family_ID is restricted, so a held-out subject's twin may sit in
training. See plan section 6.2.

CONTROL
-------
``--random-init`` trains an identically-shaped model from scratch. If it matches
the pretrained one, the result shows the architecture works, not that pretraining
transferred -- which is the actual claim. Run it at the best rung.

Usage:
    python src/02_finetuning/finetune_hcp.py --rungs last_block --device cuda
    python src/02_finetuning/finetune_hcp.py --rungs last_block all --device cuda
    python src/02_finetuning/finetune_hcp.py --rungs all --random-init --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy import stats
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing"))
from common import load_config, resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402

RUNS = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
class WindowSet(torch.utils.data.Dataset):
    """Every disjoint window of every run of the given subjects.

    Arrays are held in RAM (~2 MB per run, ~7.5 GB for the cohort) because the
    alternative is re-reading them from a network volume on every epoch.
    """

    def __init__(self, subjects, y, ts_dir, n_time, gain):
        self.n_time, self.gain = n_time, gain
        self.y = np.asarray(y, dtype=np.float32)
        self.arrays, self.index = [], []
        for si, s in enumerate(subjects):
            for r in RUNS:
                p = ts_dir / f"{s}_{r}.npy"
                if not p.exists():
                    continue
                a = np.load(p).astype(np.float32)
                ai = len(self.arrays)
                self.arrays.append(a)
                for st in range(0, a.shape[1] - n_time + 1, n_time):
                    self.index.append((si, ai, st))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, ai, st = self.index[i]
        w = self.arrays[ai][:, st : st + self.n_time] * self.gain
        return torch.from_numpy(np.ascontiguousarray(w)), self.y[si], si


class Regressor(nn.Module):
    """Encoder + 3-layer MLP on the CLS token, per BrainLM section 3.4."""

    def __init__(self, encoder, hidden=512, dropout=0.4):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, sig, xyz, noise):
        h = self.encoder(signal_vectors=sig, xyz_vectors=xyz, noise=noise,
                         return_dict=True).last_hidden_state
        return self.head(h[:, 0]).squeeze(-1)


def set_trainable(model, rung: str) -> int:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    if rung == "last_block":
        for p in model.encoder.encoder.layer[-1].parameters():
            p.requires_grad = True
    elif rung == "all":
        for p in model.parameters():
            p.requires_grad = True
    elif rung != "head":
        raise ValueError(f"unknown rung {rung}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_encoder(mcfg, random_init: bool, seed: int, device: str):
    sys.path.insert(0, str(resolve(mcfg["code_path"])))
    from brainlm_mae.modeling_brainlm import BrainLMForPretraining

    ckpt = resolve(mcfg["checkpoint_path"])
    if random_init:
        from brainlm_mae.configuration_brainlm import BrainLMConfig
        torch.manual_seed(seed)
        m = BrainLMForPretraining(BrainLMConfig.from_pretrained(ckpt))
    else:
        m = BrainLMForPretraining.from_pretrained(ckpt)
    ratio = float(mcfg["embedding"]["mask_ratio"])
    m.config.mask_ratio = ratio
    m.vit.embeddings.config.mask_ratio = ratio
    return m.vit.to(device)


def identity_noise(n_tokens, batch, device):
    """Monotonic noise makes BrainLM's argsort shuffle the identity, so token
    order is the parcel-major order the tokens were built in. Without it the
    Nystromformer encoder -- whose landmarks are means over CONSECUTIVE tokens --
    gives a different answer on every call."""
    return (torch.arange(n_tokens, device=device).float() / n_tokens)\
        .unsqueeze(0).expand(batch, -1).contiguous()


@torch.no_grad()
def predict_subjects(model, loader, xyz1, n_tokens, device, n_subj):
    """Per-window predictions averaged within subject."""
    model.eval()
    tot = np.zeros(n_subj, np.float64)
    cnt = np.zeros(n_subj, np.int64)
    for sig, y, si in loader:
        sig = sig.to(device, non_blocking=True)
        b = sig.shape[0]
        p = model(sig, xyz1.expand(b, -1, -1), identity_noise(n_tokens, b, device))
        p = p.float().cpu().numpy()
        for k, s in enumerate(si.numpy()):
            tot[s] += p[k]; cnt[s] += 1
    return tot / np.maximum(cnt, 1)


def r2_of(y, p):
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def run_fold(args, mcfg, ts_dir, subs_tr, y_tr, subs_va, y_va, subs_te, y_te,
             rung, seed, xyz1, n_tokens):
    torch.manual_seed(seed)
    enc = build_encoder(mcfg, args.random_init, seed, args.device)
    model = Regressor(enc, dropout=args.dropout).to(args.device)
    n_train_p = set_trainable(model, rung)

    gain, n_time = float(mcfg["input"]["gain"]), int(mcfg["input"]["window"])
    mk = lambda subs, y, shuffle: torch.utils.data.DataLoader(
        WindowSet(subs, y, ts_dir, n_time, gain), batch_size=args.batch_size,
        shuffle=shuffle, num_workers=args.workers, pin_memory=True, drop_last=False)
    dl_tr, dl_va, dl_te = mk(subs_tr, y_tr, True), mk(subs_va, y_va, False), mk(subs_te, y_te, False)

    enc_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and not n.startswith("head.")]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": args.head_lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.encoder_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    lossf = nn.MSELoss()

    best, best_state, bad, curve = -np.inf, None, 0, []
    for ep in range(1, args.max_epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for sig, y, _ in dl_tr:
            sig = sig.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            b = sig.shape[0]
            p = model(sig, xyz1.expand(b, -1, -1), identity_noise(n_tokens, b, args.device))
            loss = lossf(p, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item(); nb += 1
        pv = predict_subjects(model, dl_va, xyz1, n_tokens, args.device, len(subs_va))
        r2v = r2_of(np.asarray(y_va), pv)
        curve.append({"epoch": ep, "train_mse": tot / max(nb, 1), "val_r2": r2v,
                      "secs": round(time.time() - t0, 1)})
        print(f"      ep{ep:>2} train_mse {tot/max(nb,1):.4f}  val_R2 {r2v:+.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if r2v > best + args.min_delta:
            best, bad = r2v, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"      early stop at epoch {ep} (best val R2 {best:+.4f})", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pt = predict_subjects(model, dl_te, xyz1, n_tokens, args.device, len(subs_te))
    del model, enc
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return pt, best, curve, n_train_p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya")
    ap.add_argument("--model", default="brainlm")
    ap.add_argument("--target", default=None)
    ap.add_argument("--rungs", nargs="+", default=["last_block"],
                    choices=["head", "last_block", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--limit-subjects", type=int, default=None, help="smoke test")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    with (ROOT / "configs" / "models" / f"{args.model}.yaml").open() as fh:
        mcfg = yaml.safe_load(fh)
    target = args.target or cfg["label_column"]

    ts_dir = resolve(cfg["processed_path"]) / "timeseries"
    cohort = [l.strip() for l in open(resolve(cfg["cohort"]["subject_list"])) if l.strip()]
    pheno = pd.read_csv(resolve(cfg["metadata"]["phenotype"]), low_memory=False)
    pheno["Subject"] = pheno["Subject"].astype(str)
    tgt = pheno.set_index("Subject")[target]
    subs = [s for s in cohort
            if pd.notna(tgt.get(s, np.nan)) and (ts_dir / f"{s}_{RUNS[0]}.npy").exists()]
    if args.limit_subjects:
        subs = subs[: args.limit_subjects]

    y_raw = np.array([tgt[s] for s in subs], float)
    lo, hi = y_raw.min(), y_raw.max()
    y = ((y_raw - lo) / (hi - lo)).astype(np.float32)   # BrainLM section 3.4
    print(f"target {target}: {len(subs)} subjects, raw [{lo:g},{hi:g}], "
          f"scaled var {y.var():.4f}")
    print(f"device {args.device} | batch {args.batch_size} | rungs {args.rungs} | "
          f"{'RANDOM INIT' if args.random_init else 'pretrained'}")

    n_parcels = 424
    n_tokens = n_parcels * (int(mcfg["input"]["window"]) // 20)
    xyz = np.loadtxt(resolve(mcfg["input"]["coordinates"]))[:, 1:].astype(np.float32)
    if mcfg["input"]["coordinate_units"] == "unit":
        xyz = (xyz - xyz.min(0)) / (xyz.max(0) - xyz.min(0))
    xyz1 = torch.from_numpy(xyz).unsqueeze(0).to(args.device)

    rows, preds, curves = [], [], {}
    for rung in args.rungs:
        for seed in args.seeds:
            oof = np.full(len(subs), np.nan)
            kf = KFold(args.folds, shuffle=True, random_state=seed)
            t_rung = time.time()
            for fi, (tr, te) in enumerate(kf.split(np.arange(len(subs))), 1):
                rng = np.random.default_rng(seed * 100 + fi)
                perm = rng.permutation(tr)
                nv = max(1, int(len(perm) * args.val_frac))
                va, tr2 = perm[:nv], perm[nv:]
                print(f"    {rung} seed{seed} fold{fi}: "
                      f"train {len(tr2)} val {len(va)} test {len(te)}", flush=True)
                pt, bv, curve, npar = run_fold(
                    args, mcfg, ts_dir,
                    [subs[i] for i in tr2], y[tr2],
                    [subs[i] for i in va], y[va],
                    [subs[i] for i in te], y[te],
                    rung, seed, xyz1, n_tokens)
                oof[te] = pt
                curves[f"{rung}_s{seed}_f{fi}"] = curve
            r2 = r2_of(y, oof)
            r = float(stats.pearsonr(y, oof)[0])
            mse = float(((y - oof) ** 2).mean())
            mins = (time.time() - t_rung) / 60
            rows.append({"rung": rung, "seed": seed, "init":
                         "random" if args.random_init else "pretrained",
                         "trainable_params": npar, "r2": r2, "r": r, "mse": mse,
                         "minutes": round(mins, 1)})
            preds.append(pd.DataFrame({"subject_id": subs, "y_true": y, "y_pred": oof,
                                       "rung": rung, "seed": seed}))
            print(f"  == {rung} seed{seed}: out-of-fold R2 {r2:+.4f}  r {r:+.4f}  "
                  f"MSE {mse:.4f}  ({mins:.0f} min)", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== fine-tuning ladder, " + target + " ===")
    print(df.to_string(index=False))
    print("\nfrozen reference (plan 4.1): R2 +0.0748  r +0.2841")

    tag = "_randominit" if args.random_init else ""
    out = ROOT / "results" / args.dataset / args.model / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"finetune_ladder_{target}{tag}.csv", index=False)
    pd.concat(preds).to_csv(out / f"finetune_predictions_{target}{tag}.csv", index=False)
    (out / f"finetune_curves_{target}{tag}.json").write_text(json.dumps(curves, indent=1))
    stamp(out, "src/02_finetuning/finetune_hcp.py", ROOT, target=target,
          rungs=args.rungs, seeds=args.seeds, folds=args.folds,
          random_init=args.random_init, n_subjects=len(subs),
          best_r2=float(df.r2.max()))
    print(f"\n-> {out}/finetune_ladder_{target}{tag}.csv")


if __name__ == "__main__":
    main()
