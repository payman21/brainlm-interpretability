# 01_preprocessing

Scripts take `--dataset <name>` and read `configs/datasets/<name>.yaml`. No
dataset-specific constants live in the code.

## How the HCP-YA parcels in this study were actually produced

Not by `parcellate.py`, and it could not have done it. That script resolves one
BOLD file per subject from a local `raw_path`, while HCP-YA is four runs per
subject streamed from S3 and deleted after verification — 4.02 TB of source
NIfTI that was never staged on a filesystem. The cluster pipeline in
`src/00_acquisition/cluster/` did the work: fetch four runs, parcellate, verify,
delete, one Slurm array task per subject.

Its output was checked bit-for-bit against `parcellate.py` on two subjects, so
the generic path below documents the method even though it did not run it.

```bash
# 1. Parcellation on the cluster — the only voxel-level pass.
#    Set HCP_ROOT to the cluster scratch directory first.
sbatch src/00_acquisition/cluster/process_subject.sh
python  src/00_acquisition/cluster/audit.py          # verify the cohort is sound

# 2. Scaling — robust per subject per parcel, runs off cached parcels (~minutes)
python src/01_preprocessing/scale_hcp_timeseries.py --dataset hcp_ya
```

`scale_hcp_timeseries.py` is deliberately much smaller than
`prepare_timeseries.py`. HCP-YA needs no nuisance regression (the source is
already ICA-FIX cleaned, the same denoising family BrainLM pretrained on), no
resampling (TR 0.72 s against a pretraining 0.735 s, a 2.04% difference uniform
across a single-site single-protocol cohort) and no scrub-filling. The config
records each of those as an explicit decision rather than an omission.

The generic multi-site path — `build_manifest.py`, `parcellate.py`,
`prepare_timeseries.py` — is retained because it defines the method and because
the atlas handling in `parcellate.py` is what the cluster pipeline reimplements.

Parcellation requires the A424 atlas — see `external/atlases/README.md`.

## Why parcellation comes first

Nuisance regression, scrub-filling and polyphase resampling are all linear along
time and identical across voxels, so they commute with the spatial mean that
parcellation performs: applying them to 424 parcel time series gives the same
result as applying them to ~200,000 voxels. Doing it in this order turns
terabytes into ~400 KB per subject, and means the later steps can be re-run with
different settings in minutes without touching the source volumes.

The source volumes are read exactly once.

## Outputs

| Path | Contents |
|---|---|
| `$HCP_ROOT/parcels/<sub>_<run>.npy` | raw parcel time series, native TR |
| `$HCP_ROOT/motion/<sub>/` | movement regressors per run |
| `data/HCP_young_adults/processed/timeseries/<sub>.npy` | scaled, model-ready |

## Decisions encoded here

All of these are set in `configs/datasets/hcp_ya.yaml`, with the rationale
beside them. Change them together.

- **Robust scaling, per subject per parcel.** A known divergence from BrainLM,
  which computed median/IQR across subjects per parcel. Input gain 7 compensates,
  confirmed by a reconstruction-R² sweep on HCP.
- **The frame floor is applied to NATIVE frames**, not resampled ones.
  Resampling inflates the timepoint count without adding information.
- **One QC exclusion, label-free and fixed before any target was modelled.**
  `QC_Issue` code C, an acquisition fault in the signal itself, drops 78 of 1,016
  subjects. Checked for target bias before applying.
- **No motion-based exclusion applied downstream.** The threshold
  (`cohort.min_usable_frames`) is fixed in the config, and must not be re-tuned
  after seeing outcomes.
- **Parcel 405 is zeroed.** "Vermis Crus I" is a single voxel in the 2 mm atlas
  against a median parcel of 252; zeroing keeps the input 424-channel with that
  parcel contributing nothing, which is how BrainLM's toolkit treats missing.
