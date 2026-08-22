"""Final audit: open every parcel file and confirm the cohort is sound."""
import json, collections, glob, os
import numpy as np

ROOT = os.environ["HCP_ROOT"]  # cluster scratch directory
RUNS = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]
subs = [l.strip() for l in open(f"{ROOT}/hcp_subjects.txt") if l.strip()]

bad, T_by_run, windows, dropout = [], {}, 0, {}
gm = []
for s in subs:
    for r in RUNS:
        p = f"{ROOT}/parcels/{s}_{r}.npy"
        if not os.path.exists(p):
            bad.append((s, r, "missing")); continue
        try:
            a = np.load(p)
        except Exception as e:
            bad.append((s, r, f"unreadable: {e}")); continue
        if a.shape[0] != 424:      bad.append((s, r, f"parcels={a.shape[0]}")); continue
        if a.dtype != np.float32:  bad.append((s, r, f"dtype={a.dtype}")); continue
        if not np.isfinite(a).all(): bad.append((s, r, "non-finite")); continue
        T = a.shape[1]
        T_by_run[(s, r)] = T
        windows += T // 200
        z = int((a.std(axis=1) == 0).sum())
        if z: dropout[(s, r)] = z
        gm.append(float(a.mean()))

Ts = list(T_by_run.values())
print(f"files audited      : {len(T_by_run)} / {len(subs)*4}")
print(f"failures           : {len(bad)}")
for b in bad[:10]: print("   ", b)
print(f"timepoints         : min {min(Ts)}  max {max(Ts)}  full-1200 {sum(t==1200 for t in Ts)}  short {sum(t<1200 for t in Ts)}")
print(f"total 200-TP windows: {windows}   (mean {windows/len(subs):.2f} per subject)")
print(f"grand mean of means : {np.mean(gm):.1f}  (min {min(gm):.1f}, max {max(gm):.1f})")
print(f"runs with a flat/empty parcel: {len(dropout)}")
cnt = collections.Counter(v for v in dropout.values())
print(f"  distribution of #flat parcels per affected run: {dict(cnt)}")
short_sub = sorted({s for (s, r), t in T_by_run.items() if t < 1200})
print(f"subjects with >=1 short run: {len(short_sub)}")
wp = collections.Counter(sum(T_by_run[(s,r)]//200 for r in RUNS) for s in subs)
print(f"windows-per-subject distribution: {dict(sorted(wp.items()))}")
json.dump({"n_runs": len(T_by_run), "failures": bad, "total_windows": windows,
           "short_subjects": short_sub,
           "dropout_runs": {f"{s}_{r}": v for (s, r), v in dropout.items()}},
          open(f"{ROOT}/audit_report.json", "w"), indent=1)
print("\nwrote audit_report.json")
