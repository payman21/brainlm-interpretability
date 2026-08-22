# Sparse-feature attribution on an fMRI foundation model

Code and derived result tables for the study described in the accompanying
submission: what does a brain foundation model's prediction actually run on?

BrainLM (`old_13M`, a masked autoencoder pretrained on 6,700 h of resting-state
fMRI) is applied frozen to the Human Connectome Project Young Adult cohort. Two
independent analyses ask what its prediction of fluid intelligence relies on — a
supervised distillation into pre-specified signal features, and an unsupervised
attribution over a sparse-autoencoder dictionary. Both converge on lag-1
autocorrelation in the central executive network.

Everything here is anonymised for double-blind review.

## Reproduce the figures and tables in five minutes

Every figure panel reads a committed CSV, so the numbers in the paper can be
checked without HCP access, a GPU, or the model checkpoint:

```bash
pip install numpy pandas scipy scikit-learn matplotlib
python figures/make_figures.py
```

That writes six PDFs into `figures/`. The tables they read are under
`results/hcp_ya/brainlm/`.

## Repository layout

```
configs/datasets/hcp_ya.yaml     every preprocessing and cohort decision, with rationale
configs/models/brainlm.yaml      checkpoint, input gain, coordinate mode, pooling
external/atlases/A424/           the parcellation BrainLM was pretrained on
src/00_acquisition/              phenotype export; the cluster pipeline that built the parcels
src/01_preprocessing/            parcellation and scaling
src/02_finetuning/               embedding extraction, input-scale check, fine-tuning ladder
src/03_evaluation/               frozen prediction and its controls
src/04_interpretability/         probing, SAE, attribution, distillation
figures/make_figures.py          every figure, from the committed tables
results/hcp_ya/                  aggregate result tables (see below)
```

## Pipeline order

The held-out split is drawn first, before any interpretability step, and every
downstream result is conditioned on it.

```bash
python src/04_interpretability/split_holdout_hcp.py    --dataset hcp_ya   # 656 train / 280 holdout
python src/02_finetuning/extract_embeddings.py         --dataset hcp_ya --model brainlm
python src/03_evaluation/predict_frozen_hcp.py         --dataset hcp_ya
python src/03_evaluation/frozen_controls_hcp.py        --dataset hcp_ya
python src/04_interpretability/probe_signal_features_hcp.py --dataset hcp_ya
python src/04_interpretability/extract_token_acts_hcp.py    --dataset hcp_ya
python src/04_interpretability/train_sae_hcp.py        --dataset hcp_ya \
       --subjects-file results/hcp_ya/splits/train_subjects.txt
python src/04_interpretability/encode_sae_windows_hcp.py    --dataset hcp_ya
python src/04_interpretability/sae_attribution_hcp.py       --dataset hcp_ya
python src/04_interpretability/characterise_sae_hcp.py      --dataset hcp_ya
python src/04_interpretability/distill_hcp.py               --dataset hcp_ya
python src/04_interpretability/distill_confirm_hcp.py       --dataset hcp_ya
```

Two things that are easy to get wrong and that the scripts guard against:

- **`--subjects-file` on the SAE is not optional.** A dictionary fitted on all
  936 subjects has seen the holdout, and every attribution built on it inherits
  that contact. Unsupervised is not the same as uninvolved.
- **`split_holdout_hcp.py` refuses to overwrite without `--force`.** Re-drawing
  the split invalidates every number downstream of it.

## What is here, and what is not

**Included** — everything keyed by target, feature, network or parcel:

| Path | Contents |
|---|---|
| `results/hcp_ya/brainlm/metrics/` | frozen prediction, controls, fine-tuning ladder, distillation, holdout confirmation |
| `results/hcp_ya/brainlm/interpretability/` | signal probing, pretraining gain, token-identity probes |
| `results/hcp_ya/brainlm/interpretability/sae/` | attribution (cumulative, controls, summary, per-feature) and feature characterisation |
| `results/hcp_ya/splits/` | the split manifest, and the train/holdout subject lists |

**Not included:**

- **Subject-level activation matrices** (`sae_features.npz`,
  `sae_window_means.npz`, ~300 MB) and per-subject prediction tables. These are
  subject-level derived data from a consortium-governed dataset. They are
  regenerable from the code above given HCP access.
- **HCP-YA source data.** Available from the open-access tier under the WU-Minn
  HCP Open Access Data Use Terms, which must be accepted before download.
- **The BrainLM checkpoint and toolkit.** Publicly released by its authors; not
  redistributed here.
- **The trained SAE weights** (`sae.pt`, 16 MB) — hosted separately, see the
  data availability statement in the submission.

The split lists are included because the holdout claim cannot be checked without
them. They are open-access HCP-YA subject identifiers and carry no phenotype;
the split is stratified by target quintile, so membership is balanced by
construction rather than informative about any individual.

## Known gaps

- `parcel_state_sae_hcp.py` writes `sae_parcel_state.csv`, which is not in this
  release — the script's output path changed after the reported run and it has
  not been re-executed. Nothing in the paper's figures or tables reads it.
- The nonlinear atlas warp path in `parcellate.py` is untested; it was not
  needed here, since the HCP-YA grid is bit-identical to the atlas.
- Folds are subject-level, not family-level. HCP-YA is a twin study and
  `Family_ID` is restricted-access, so a held-out subject's twin may be in
  training. This is reported as a limitation in the paper rather than worked
  around.

## Licence

Code in this repository: MIT (see `LICENSE`). The A424 atlas under
`external/atlases/` is redistributed from its own source under its own terms —
see `external/atlases/A424/README.md`. HCP-YA data are governed by the WU-Minn
HCP Open Access Data Use Terms and are not redistributed here.
