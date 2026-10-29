# ULF DTI geometric distortion and bias field correction

Joint angularly dependent distortion and bias correction for low-field DTI using Beta FA/DSW V1 atlas likelihoods and tensor consistency loss. CPU only (GPU to come :) ).

## Setup

Install the Python dependencies in `requirements.txt`.
A CPU PyTorch build is sufficient. Also put these in `PATH` if you haven't already done so:

- ANTs: `antsRegistration`, `antsApplyTransforms`, `ConvertTransformFile`, `N4BiasFieldCorrection`.
- MRtrix3: `mrconvert`, `mrcat`, `mrmath`, `dwidenoise`.
- FreeSurfer: `mri_synthsr`, `mri_synthstrip`, `mri_super_synth`, `mri_convert`.
- A C++ compiler with OpenMP, normally `g++`.

Use a terminal where FreeSurfer is already working, with its license installed.
No manual compilation is needed; `run.sh` builds the image sampler automatically.

## Full run


```bash
bash run.sh all \
    --dwi /path/to/full_sequence.nii.gz \
    --bvals /path/to/bvals.txt \
    --bvecs /path/to/bvecs.txt \
    --out /path/to/new_output \
    --threads 2
```


All that's needed is your [W,H,D,C] dMRI/DTI volume and an FSL-style bvec table (N-by-3 is also accepted). B0 volumes and DWIs can appear in any order. For linear-encoding DTI, at least one B0 and six independent tensor measurements are required. Single-shell and multi-shell inputs are supported, using each DWI's own b-value, however this remains a single-tensor model, not a general high-b/non-Gaussian diffusion estimate.


Proprocessing goes through SynthSR/SynthStrip guides, rigid + affine alignment of acquired volumes, b-vector updating, denoising, and SuperSynth segmentation. Atlas registration uses rigid + affine + SyN with segmentation-guided refinement and V1 reorientation. Low-B bias prefit then precedes joint correction.

Inspect `atlas/registration.png` before interpreting results. There is no per-volume nonlinear preprocessing, eddy, or topup. Existing output directories are not overwritten. Use `preprocess` instead of `all` to stop before correction.

## Correction WITHOUT preprocessing

```bash
bash run.sh correct \
    --dwi /path/to/affine_denoised_12vol.nii.gz \
    --bvals /path/to/bvals.txt \
    --bvecs /path/to/rotated_bvecs.txt \
    --mask /path/to/brain_mask.nii.gz \
    --seg /path/to/segmentation.nii.gz \
    --atlas /path/to/registered_priors \
    --out /path/to/new_output \
    --threads 2
```

The registered-prior directory needs:

```text
alpha_deformed_resampled.nii.gz
beta_deformed_resampled.nii.gz
kappa_deformed_resampled.nii.gz
v1mu_deformed_resampled.nii.gz
deformed_mni_mask.nii.gz
```

Inputs must share the common grid and affine transform. Atlas DSW V1 components must match the tensor-fit gradient convention. `--check` validates inputs without fitting.

## Options and defaults

- `--grid 20|30|40`: spacing in mm for both distortion and bias grids; default 30.
- `--b0-threshold 50`: b-values at or below this threshold are averaged as the
  low-B reference (s/mm²). These are treated as approximately unweighted signal.
- `--fa-tissue-split`: use GM/WM-specific Beta priors; off by default. DSW is
  unchanged. Prepared inputs also need registered `alpha_gm/wm`, `beta_gm/wm`,
  and `mask_gm/wm` maps with the `_deformed_resampled.nii.gz` suffix.
- `--skip-b0-prefit`: skip the separate low-B bias fit, e.g. for already-prefitted data.
- `--fixed-b0`: disable low-B deformation. Otherwise all low-B frames share
  one learned field independent of the per-DWI fields.

For `all`, pass other correction arguments after `--`, e.g. `-- --cycles 50`.
For synthetic data with untouched low-B frames, use both
`--skip-b0-prefit` and `--fixed-b0`.

| Setting | Default |
|---|---:|
| Distortion / bias spacing | 30 / 30 mm |
| Maximum displacement norm | 20 mm |
| Cycles | 300 |
| Updates per cycle | 1 bias + 5 distortion |
| Distortion optimizer / LR | SGD, no momentum / 300 |
| Bias optimizer / LR | Adam / 0.01 |
| Distortion / bias gradient clipping | 1 / 5 |
| Tensor consistency weight | 400 |
| Beta FA weight | 0.05 |
| DSW V1 weight | 0.2 |
| Bias amplitude weight | 0.2 |
| Displacement weight | 0.0025 |
| Bending weight | 0.625 |
| Common-motion weight | 0.02 |

Tensor consistency is evaluated in WM by default. DSW V1 uses `kappa * subject FA` in both the orientation likelihood and Kummer normalization. Bias has a hard voxelwise directional mean-zero constraint. Regularizers penalize dense log-bias amplitude, displacement magnitude, bending, and directional-mean DWI displacement. Sampling uses prefiltered tricubic interpolation (see cubic_pull.cpp) in intensity space.

The separate B0 fit estimates a polynomial log-gain by matching ADC distributions across eroded cortical GM, peripheral WM, thalamus, and brainstem masks. The gain is applied to each individual low-B frame, not the DWIs. This tissue-distribution assumption CN alter regional ADC differences, and does not uniquely identify physical receive bias.

## Outputs

- `final/full_sequence.joint_corrected.nii.gz`: all individual corrected B0
  frames and DWIs in input order, with `full_sequence.bval` and `.bvec` sidecars.
- `dmri.joint_corrected.nii.gz`: DWI frames only, with `dmri.bval` and `.bvec`.
- FA, ADC, V1, color FA, mean low-B/DWI, eigenvalues, and geometry-only component data.
- `joint_dwi_displacement_field.nii.gz`: `[X,Y,Z,3,Ndwi]`, XYZ displacement in mm.
- `b0_displacement_field.nii.gz`: `[X,Y,Z,3,1]`, shared low-B displacement.
- `joint_log_bias_field.nii.gz`: `[X,Y,Z,Ndwi]`, dimensionless stabilized log bias.
- `volume_indices.json`: zero-based input indices corresponding to the B0 and
  DWI frames; per-DWI field axes follow this DWI order.
- Jacobian, invalid-sampling, and negative eigenvalue QC.
- `configuration.json`, `loss.csv`, selected `parameters.pt`, and latest `checkpoint.pt`.


## OPTIONAL atlas building

To build atlas priors:

```bash
HCP_ROOT=/path/to/training_subjects \
TARGET=/path/to/mni_t2_masked.nii.gz \
OUT=/path/to/new_atlas JOBS=2 THREADS=2 bash atlas/build.sh
```

This also requires NiftyReg, MRtrix `dwiextract`, `mrgrid`, `dwi2tensor`,
`tensor2metric`, and FreeSurfer `mri_synthseg`. Default inputs are
`subject*/data.nii.gz`, `bvals`, and `bvecs`; override `SUBJECT_GLOB`, `DWI_REL`,
`BVALS_REL`, or `BVECS_REL` as needed. Note that the builder does not produce tissue-split
priors. Verify gradient/vector conventions for new acquisitions or atlases.


