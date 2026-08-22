# A424

The parcellation BrainLM was pretrained on: 424 regions covering cortex,
subcortex and cerebellum.

**Source:** <https://github.com/emergelab/hierarchical-brain-networks>
(`brainmaps/`), the repository for Akiki & Abdallah, *Determining the
Hierarchical Architecture of the Human Brain Using Subject-Level Clustering of
Functional Networks*, Scientific Reports 9, 19290 (2019),
doi:10.1038/s41598-019-55738-y. Redistributed here unmodified, for
reproducibility; please cite the original.

## Files

| File | Notes |
|---|---|
| `A424+2mm.nii.gz` | the volume in use — cortical GM expanded 2 mm into WM, 424 labels |
| `A424_Coordinates.dat` / `.txt` | parcel centroid coordinates, read as BrainLM's spatial input |
| `A424_Labels_AA-AAc_main_maps.csv` | parcel labels and AA-7 network assignments |

The upstream repository also ships `A424.nii.gz` and `A424+4mm.nii.gz`, not
copied here because neither is used.

## Two things to know

**1. `+2mm` is dilation, not resolution.** All the upstream volumes are
91×109×91 at 2 mm isotropic. The suffix denotes how far cortical grey matter was
expanded into white matter. `+2mm` is used here because the undilated
`A424.nii.gz` is **missing label 120 entirely** — it has 423 parcels, not 424,
which would silently shift every downstream parcel index — and because it is
what the authors' own extraction script (`Get_A424_TS.m`) uses.

**2. No warp was applied in this study.** A424 is in `MNI152NLin6Asym` (the FSL
MNI152 grid). HCP-YA `MNINonLinear` volumes are 91×109×91 at 2 mm with affine
`[[-2,0,0,90],[0,2,0,-126],[0,0,2,-72]]` — verified bit-identical to
`A424+2mm`. No warp, no interpolation, no resampling, and
`atlas.warp_to_bold_space` is `false` in the config accordingly.

For datasets whose derivatives are in `MNI152NLin2009cAsym` this does not hold:
the two templates differ nonlinearly by a few millimetres, largest near cortical
edges and inferior structures. `parcellate.py` handles that case via
`atlas.warp_to_bold_space`, which needs `templateflow` and `nitransforms`
installed; without them it falls back to affine-only alignment, warns, and
records which path it took. **That path is untested** — it was never needed here.
