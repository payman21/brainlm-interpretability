#!/usr/bin/env python3
"""E2b -- the fine-tuning ladder. Runs on CUDA, MPS or CPU.

Frozen BrainLM features fail three preconditions: non-determinism (fixed, A7),
subject identification at chance where a correlation matrix reaches 0.78, and
CN/AD/FTD at chance. The open question is whether FINE-TUNING rescues the model.
Until that is answered the pre-registered gate is unresolved and "you only probed
frozen features" is a free objection.

DESIGN (amendment A6, revised by A8 before any E2b result was seen)
-------------------------------------------------------------------
Splits are by SITE, so no result can be explained by site leakage:

    train  LO, MA, TA          (459)
    val    RE, AF              ( 84)   site-grouped, early stopping only
    test1  MI/0.85             (138)   PRE-EXPOSED -- see below
    test2  SL, BE              (159)   never touched by any analysis

**test1 is not a pristine held-out set and is not described as one.** MI/0.85 was
used to select the input gain (label-free), and to run the frozen classification
and reliability analyses (labelled). It is reported as a *pre-exposed target
domain*. test2 has been used for nothing and is the clean generalisation estimate.
Both are reported; neither is selected after the fact.

Every baseline is trained and evaluated on EXACTLY these subjects. This matters:
age+sex scores 0.563 within MI by cross-validation but **0.335 when trained on
other sites and applied to MI** -- the demographic confound does not transfer.
Comparing a cross-site model against the within-site number is not defensible.

THE GATE (pre-committed)
------------------------
A foundation-model claim requires BOTH:
    (a) test CI lower bound above the age+sex baseline **on the same split**, and
    (b) pretrained initialisation beating its random-init twin, consistently
        across seeds -- not on one favourable draw.
(b) exists because (a) alone can be passed by an architecture that never saw
pretraining.

REPORTING (pre-committed)
-------------------------
Underpowered by design. Three seeds per configuration, full per-seed table
reported rather than the best rung. Subject-level predictions are saved so paired
comparisons, confusion matrices and per-class recall can be computed afterwards
without re-running.

Usage:
    python src/02_finetuning/finetune_ladder.py --device cuda
    python src/02_finetuning/finetune_ladder.py --device cpu --smoke
"""

from __future__ import annotations

import argparse, json, platform, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "01_preprocessing")); sys.path.insert(0, str(ROOT / "src"))
from common import load_config, resolve  # noqa: E402
from common_provenance import stamp  # noqa: E402

RESULTS = ROOT / "results"  # dataset subdirectory is appended at write time
CLASSES = ["CN", "AD", "FTD"]
SPLIT_SITES = {"train": ["LO", "MA", "TA"], "val": ["RE", "AF"], "test2": ["SL", "BE"]}


def identity_noise(model, batch, device):
    """Monotonic noise so BrainLM's argsort shuffle is the identity (A7)."""
    n = model.config.num_brain_voxels * (
        model.config.num_timepoints_per_voxel // model.config.timepoint_patching_size)
    return (torch.arange(n, dtype=torch.float32, device=device) / n).repeat(batch, 1)


def eval_starts(n_frames: int, n_time: int, want: int) -> np.ndarray:
    """Non-overlapping evaluation windows, as many as the run supports.

    An earlier version spaced `want` windows evenly regardless of run length, which
    silently produced OVERLAPPING windows for 63/137 validation subjects while the
    docstring claimed they were disjoint. Short runs now simply contribute fewer
    windows.
    """
    k = max(1, min(want, n_frames // n_time))
    return (np.linspace(0, n_frames - n_time, k).astype(int) if k > 1 else np.array([0]))


class Windows(torch.utils.data.Dataset):
    def __init__(self, ids, labels, ts_dir, n_time, gain, train, want=3, seed=0):
        self.ids, self.labels, self.train = list(ids), np.asarray(labels), train
        self.n_time, self.gain = n_time, gain
        self.data = [np.load(ts_dir / f"{s}.npy").astype(np.float32) for s in self.ids]
        self.rng = np.random.default_rng(seed)
        if not train:
            self.index = [(i, s) for i, d in enumerate(self.data)
                          for s in eval_starts(d.shape[1], n_time, want)]

    def __len__(self):
        return len(self.ids) if self.train else len(self.index)

    def __getitem__(self, i):
        if self.train:
            ts = self.data[i]
            s = int(self.rng.integers(0, max(1, ts.shape[1] - self.n_time + 1)))
            return torch.from_numpy(ts[:, s:s + self.n_time] * self.gain), int(self.labels[i]), i
        subj, s = self.index[i]
        ts = self.data[subj]
        return torch.from_numpy(ts[:, s:s + self.n_time] * self.gain), int(self.labels[subj]), subj


class Classifier(nn.Module):
    """Encoder + head. The 3-layer MLP matches BrainLM's published downstream
    protocol for clinical-variable prediction; linear is kept for comparison."""

    def __init__(self, encoder, hidden, n_classes, kind="mlp", dropout=0.4):
        super().__init__()
        self.encoder = encoder
        self.head = (nn.Linear(hidden, n_classes) if kind == "linear" else
                     nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(hidden // 2, n_classes)))

    def forward(self, sig, xyz, noise):
        h = self.encoder(signal_vectors=sig, xyz_vectors=xyz, noise=noise,
                         return_dict=True).last_hidden_state
        return self.head(h[:, 0])


def set_trainable(model, rung):
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
        raise ValueError(rung)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict(model, loader, xyz, device, n_sub):
    """Subject-level: mean logits over that subject's windows."""
    model.eval()
    logits = np.zeros((n_sub, len(CLASSES))); cnt = np.zeros(n_sub); truth = np.full(n_sub, -1)
    for sig, y, idx in loader:
        sig = sig.to(device)
        out = model(sig, xyz[:sig.shape[0]], identity_noise(model.encoder, sig.shape[0], device))
        for j, s in enumerate(idx.numpy()):
            logits[s] += out[j].float().cpu().numpy(); cnt[s] += 1; truth[s] = y[j]
    logits /= np.maximum(cnt, 1)[:, None]
    return truth, logits, cnt


def boot_ci(y, yhat, n=2000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            out.append(balanced_accuracy_score(y[i], yhat[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def summarise(tag, truth, pred, extra):
    lo, hi = boot_ci(truth, pred)
    cm = confusion_matrix(truth, pred, labels=range(len(CLASSES)))
    adftd = (truth != 0)
    rec = {**extra, "eval_set": tag, "n": int(len(truth)),
           "balanced_accuracy": round(balanced_accuracy_score(truth, pred), 4),
           "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
           "confusion_matrix": cm.tolist(),
           "per_class_recall": [round(v, 4) for v in
                                (cm.diagonal() / np.maximum(cm.sum(1), 1)).tolist()]}
    if adftd.sum() > 1 and len(np.unique(truth[adftd])) > 1:
        rec["ad_vs_ftd_bal_acc"] = round(
            balanced_accuracy_score(truth[adftd], np.clip(pred[adftd], 1, 2)), 4)
    return rec


def baselines(splits, coh, proc, seed=0):
    """Returns (summary rows, per-subject predictions) on EXACTLY the ladder's split."""
    """age+sex and Fisher-z FC, trained and tested on EXACTLY the ladder's split."""
    out, per = [], {}
    tr = coh.set_index("subject_id").loc[splits["train"][0]]
    ytr = splits["train"][1]
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(tr[["sex"]].astype(str))
    dem = lambda d: np.hstack([np.nan_to_num(d[["age"]].to_numpy(float), nan=float(tr.age.mean())),
                               enc.transform(d[["sex"]].astype(str))])

    # ONE parcel mask for every subject. Selecting std>0 per subject keeps a
    # different number of parcels each time, giving ragged feature vectors; and a
    # parcel that is flat in one subject returns NaN from corrcoef, which silently
    # destroys the whole vector. Parcels flat in ANY subject are dropped from all.
    # This is a data-availability property, not label information.
    fc_mask = None
    for s in sorted({i for k in splits for i in splits[k][0]}):
        sd = np.load(proc / "timeseries" / f"{s}.npy")[:, :200].std(1) > 0
        fc_mask = sd if fc_mask is None else (fc_mask & sd)

    def fc(ids):
        V = []
        for s in ids:
            ts = np.load(proc / "timeseries" / f"{s}.npy")[fc_mask][:, :200]
            r = np.corrcoef(ts); iu = np.triu_indices_from(r, 1)
            V.append(np.arctanh(np.clip(r[iu], -.999, .999)))
        return np.asarray(V, dtype=np.float32)

    for name, feat in [("age_sex", lambda ids: dem(coh.set_index("subject_id").loc[ids])),
                       ("fc_fisherz", fc)]:
        Xtr = feat(splits["train"][0])
        pipe = Pipeline([("sc", StandardScaler()),
                         ("pca", PCA(min(64, Xtr.shape[1], len(Xtr) - 1), random_state=seed)),
                         ("e", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        pipe.fit(Xtr, ytr)
        for tag in [k for k in splits if k.startswith("test")]:
            ids, y = splits[tag]
            p = pipe.predict(feat(ids))
            out.append(summarise(tag, np.asarray(y), p,
                                 {"rung": name, "init": "-", "seed": seed}))
            per[(name, tag)] = p
    return out, per


def run_one(rung, init, seed, cfg, mcfg, splits, args):
    from brainlm_mae.modeling_brainlm import BrainLMForPretraining
    from brainlm_mae.configuration_brainlm import BrainLMConfig
    torch.manual_seed(seed); np.random.seed(seed)
    ckpt = resolve(mcfg["checkpoint_path"])
    # The published protocol applies 40% dropout to the activations of BOTH the
    # encoder and the MLP head during clinical fine-tuning (brainlm.md line 110).
    conf = BrainLMConfig.from_pretrained(ckpt)
    conf.hidden_dropout_prob = args.dropout
    base = (BrainLMForPretraining(conf) if init == "random"
            else BrainLMForPretraining.from_pretrained(ckpt, config=conf))
    base.config.mask_ratio = base.vit.embeddings.config.mask_ratio = 0.0
    model = Classifier(base.vit, base.config.hidden_size, len(CLASSES),
                       kind=args.head, dropout=args.dropout).to(args.device)
    n_par = set_trainable(model, rung)

    ts_dir = resolve(cfg["processed_path"]) / "timeseries"
    n_time, gain = int(mcfg["input"]["window"]), float(mcfg["input"]["gain"])
    mk = lambda tag, train: Windows(*splits[tag], ts_dir, n_time, gain, train, seed=seed)
    dl = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=sh)
    tr_ds, va_ds = mk("train", True), mk("val", False)

    c = np.loadtxt(resolve(mcfg["input"]["coordinates"]))[:, 1:].astype(np.float32)
    xyz = torch.from_numpy(c).unsqueeze(0).repeat(args.batch_size, 1, 1).to(args.device)
    counts = np.bincount(splits["train"][1], minlength=len(CLASSES))
    w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)),
                     dtype=torch.float32, device=args.device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    groups = [{"params": model.head.parameters(), "lr": args.head_lr}]
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    if enc:
        groups.append({"params": enc, "lr": args.encoder_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    best, best_state, best_ep, wait, curve = -np.inf, None, -1, 0, []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for sig, y, _ in dl(tr_ds, True):
            sig, y = sig.to(args.device), y.to(args.device)
            opt.zero_grad()
            l = loss_fn(model(sig, xyz[:sig.shape[0]],
                              identity_noise(model.encoder, sig.shape[0], args.device)), y)
            l.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); tot += float(l)
        yt, lg, _ = predict(model, dl(va_ds, False), xyz, args.device, len(splits["val"][0]))
        va = balanced_accuracy_score(yt, lg.argmax(1))
        curve.append({"epoch": ep, "train_loss": round(tot / max(1, len(tr_ds)), 4),
                      "val_bal_acc": round(va, 4), "sec": round(time.time() - t0, 1)})
        print(f"    ep {ep+1:>2}/{args.epochs} loss {curve[-1]['train_loss']:.3f} "
              f"val {va:.3f} ({curve[-1]['sec']:.0f}s)", flush=True)
        if va > best:
            best, best_ep, wait = va, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"    early stop @ {ep+1}"); break

    model.load_state_dict(best_state)
    rows, preds = [], []
    for tag in [k for k in splits if k.startswith("test")]:
        ds = mk(tag, False)
        yt, lg, cnt = predict(model, dl(ds, False), xyz, args.device, len(splits[tag][0]))
        p = lg.argmax(1)
        rows.append(summarise(tag, yt, p, {"rung": rung, "init": init, "seed": seed,
                                           "trainable_params": int(n_par),
                                           "best_epoch": best_ep, "val_bal_acc": round(best, 4)}))
        for i, sid in enumerate(splits[tag][0]):
            preds.append({"subject_id": sid, "eval_set": tag, "rung": rung, "init": init,
                          "seed": seed, "true": int(yt[i]), "pred": int(p[i]),
                          "n_windows": int(cnt[i]),
                          **{f"logit_{c}": round(float(lg[i, j]), 5) for j, c in enumerate(CLASSES)}})
        print(f"  -> {rung}/{init}/s{seed} {tag}: {rows[-1]['balanced_accuracy']:.3f} "
              f"[{rows[-1]['ci_lo']:.3f}, {rows[-1]['ci_hi']:.3f}]", flush=True)
    return rows, preds, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hcp_ya"); ap.add_argument("--model", default="brainlm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rungs", nargs="+", default=["head", "last_block", "all"])
    ap.add_argument("--inits", nargs="+", default=["pretrained", "random"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--head", default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    with (ROOT / "configs" / "models" / f"{args.model}.yaml").open() as fh:
        mcfg = yaml.safe_load(fh)
    sys.path.insert(0, str(resolve(mcfg["code_path"])))
    proc = resolve(cfg["processed_path"])
    coh = pd.read_csv(proc / "cohort.csv")
    lab = coh[coh.diagnosis.isin(CLASSES)].copy()
    lab["y"] = pd.Categorical(lab.diagnosis, categories=CLASSES).codes

    parts = {k: lab[lab.site.isin(v)] for k, v in SPLIT_SITES.items()}
    parts["test1"] = lab[lab.role == "development"]
    for st in SPLIT_SITES["test2"]:            # per-site, so a pooled score cannot
        parts[f"test2_{st}"] = lab[lab.site == st]   # hide a between-site effect
    if args.smoke:
        # Stratified, or a sorted cohort hands the smoke run a single class.
        parts = {k: v.groupby("y", group_keys=False).apply(
                    lambda g: g.head(8), include_groups=False).join(
                    v[["y"]], how="left", rsuffix="_") if False else
                 v.groupby("y", group_keys=False).head(8)
                 for k, v in parts.items()}
        args.epochs, args.seeds = 2, [0]
    splits = {k: (d.subject_id.tolist(), d.y.to_numpy()) for k, d in parts.items()}

    print(f"device={args.device} head={args.head} seeds={args.seeds}")
    for k in ["train", "val", "test1", "test2", "test2_SL", "test2_BE"]:
        d = parts[k]
        print(f"  {k:<6} n={len(d):<5} sites={sorted(d.site.unique())} "
              f"CN/AD/FTD={np.bincount(d.y, minlength=3).tolist()}")
    print("  test1 = PRE-EXPOSED target domain; test2 = never touched\n")

    rows, base_preds = baselines(splits, coh, proc)
    for r in rows:
        print(f"  baseline {r['rung']:<11} {r['eval_set']}: {r['balanced_accuracy']:.3f} "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    print()
    out = RESULTS / args.dataset / args.model / "metrics"; out.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if args.smoke else ""
    preds, curves = [], {}

    def flush():
        pd.DataFrame(rows).to_csv(out / f"e2b_ladder{tag}.csv", index=False)
        pd.DataFrame(preds).to_csv(out / f"e2b_predictions{tag}.csv", index=False)
        (out / f"e2b_curves{tag}.json").write_text(json.dumps(curves, indent=1))

    flush()
    for rung in args.rungs:
        for init in args.inits:
            for seed in args.seeds:
                try:
                    r, p, c = run_one(rung, init, seed, cfg, mcfg, splits, args)
                    rows += r; preds += p; curves[f"{rung}_{init}_{seed}"] = c
                except torch.cuda.OutOfMemoryError as e:
                    # Record and continue rather than discarding the whole ladder.
                    print(f"  !! OOM on {rung}/{init}/s{seed}: {e}", flush=True)
                    rows.append({"rung": rung, "init": init, "seed": seed,
                                 "eval_set": "FAILED", "error": "cuda_oom"})
                    torch.cuda.empty_cache()
                flush()
    df = pd.DataFrame(rows)

    # Paired difference against the demographic baseline on identical subjects --
    # the comparison that a fixed threshold cannot give.
    pr = pd.DataFrame(preds)
    pair = []
    for tag_ in ["test1", "test2"]:
        bp = base_preds[("age_sex", tag_)]
        for (rung, init, seed), g in pr[pr.eval_set == tag_].groupby(["rung", "init", "seed"]):
            g = g.reset_index(drop=True)
            m = balanced_accuracy_score(g.true, g.pred)
            b = balanced_accuracy_score(g.true, bp[:len(g)])
            pair.append({"eval_set": tag_, "rung": rung, "init": init, "seed": seed,
                         "model": round(m, 4), "age_sex": round(b, 4),
                         "delta_vs_demographics": round(m - b, 4)})
    pd.DataFrame(pair).to_csv(out / f"e2b_paired_vs_demographics{tag}.csv", index=False)

    print("\n=== summary: mean over seeds, test1 / test2 ===")
    for tag_ in ["test1", "test2"]:
        base = df[(df.eval_set == tag_) & (df.rung == "age_sex")].balanced_accuracy.iloc[0]
        g = (df[(df.eval_set == tag_) & (df.init != "-")]
             .groupby(["rung", "init"]).balanced_accuracy.agg(["mean", "min", "max"]).round(3))
        print(f"\n{tag_}  (age+sex baseline on THIS split = {base:.3f})")
        print(g.to_string())
    stamp(out, "src/02_finetuning/finetune_ladder.py", ROOT, output=f"e2b_ladder{tag}.csv",
          args=vars(args), split_sites=SPLIT_SITES,
          env={"python": platform.python_version(), "torch": torch.__version__,
               "numpy": np.__version__, "pandas": pd.__version__})
    print(f"\n-> {out / f'e2b_ladder{tag}.csv'}  (+ predictions, curves)")


if __name__ == "__main__":
    main()
