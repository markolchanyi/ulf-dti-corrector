#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
HCP_ROOT="${HCP_ROOT:?Set HCP_ROOT to the atlas-training subjects}"
TARGET="${TARGET:?Set TARGET to the masked MNI T2 template}"
OUT="${OUT:?Set OUT to a new atlas output directory}"
JOBS="${JOBS:-2}"
THREADS="${THREADS:-2}"
DWI_REL="${DWI_REL:-data.nii.gz}"
BVALS_REL="${BVALS_REL:-bvals}"
BVECS_REL="${BVECS_REL:-bvecs}"
SUBJECT_GLOB="${SUBJECT_GLOB:-subject*}"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
for exe in mrconvert dwiextract mrgrid mrmath dwi2tensor tensor2metric reg_aladin reg_resample mri_synthstrip mri_synthseg; do
    command -v "$exe" >/dev/null || { echo "Missing command: $exe" >&2; exit 2; }
done
mkdir -p "$OUT/work" "$OUT/deformed" "$OUT/data" "$OUT/fitted"
OUT="$(cd "$OUT" && pwd)"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mrgrid "$TARGET" regrid "$OUT/data/t2.nii.gz" -voxel 3
mri_synthstrip -i "$OUT/data/t2.nii.gz" -o "$OUT/work/template_stripped.nii.gz" -m "$OUT/data/mask.nii.gz"
mri_synthseg --i "$OUT/data/t2.nii.gz" --o "$OUT/data/seg.nii.gz" --robust --cpu --threads "$THREADS"

# Preserve the original affine HCP-to-template atlas-building recipe.
process_subject() {
    sdir="$1"
    subj="$(basename "$sdir")"
    tmp="$OUT/work/$subj"
    for file in "$sdir/$DWI_REL" "$sdir/$BVALS_REL" "$sdir/$BVECS_REL"; do
        [[ -s "$file" ]] || { echo "Missing input: $file" >&2; return 2; }
    done
    mkdir "$tmp"
    mrconvert "$sdir/$DWI_REL" "$tmp/dwi.mif" -fslgrad "$sdir/$BVECS_REL" "$sdir/$BVALS_REL" -nthreads "$THREADS"
    dwiextract "$tmp/dwi.mif" "$tmp/shell.mif" -shells 0,1000 -nthreads "$THREADS"
    dwiextract "$tmp/shell.mif" "$tmp/b0s.mif" -bzero -nthreads "$THREADS"
    mrmath "$tmp/b0s.mif" mean "$tmp/mean_b0.mif" -axis 3 -nthreads "$THREADS"
    mrgrid "$tmp/shell.mif" regrid "$tmp/dwi3.mif" -voxel 3 -nthreads "$THREADS"
    mrgrid "$tmp/mean_b0.mif" regrid "$tmp/b03.nii.gz" -voxel 3 -nthreads "$THREADS"
    dwi2tensor "$tmp/dwi3.mif" "$tmp/dt.mif" -nthreads "$THREADS"
    tensor2metric "$tmp/dt.mif" -fa "$tmp/fa.mif" -vector "$tmp/v1.mif" -modulate none -nthreads "$THREADS"
    mrconvert "$tmp/fa.mif" "$tmp/fa.nii.gz" -strides -1,2,3
    mrconvert "$tmp/v1.mif" "$tmp/v1.nii.gz" -strides -1,2,3
    mri_synthstrip -i "$tmp/b03.nii.gz" -o "$tmp/b03_stripped.nii.gz" --no-csf
    reg_aladin -ref "$OUT/data/t2.nii.gz" -flo "$tmp/b03_stripped.nii.gz" \
        -aff "$tmp/affine.txt" -res "$tmp/b0_registered.nii.gz" -omp "$THREADS"
    reg_resample -ref "$OUT/data/t2.nii.gz" -flo "$tmp/fa.nii.gz" -res "$OUT/deformed/$subj.fa.nii.gz" \
        -trans "$tmp/affine.txt" -inter 1 -omp "$THREADS"
    reg_resample -ref "$OUT/data/t2.nii.gz" -flo "$tmp/v1.nii.gz" -res "$tmp/v1_resampled.nii.gz" \
        -trans "$tmp/affine.txt" -inter 0 -omp "$THREADS"
    "$PYTHON" "$DIR/rotate.py" --aff "$tmp/affine.txt" --in-v1 "$tmp/v1_resampled.nii.gz" \
        --out-v1 "$OUT/deformed/$subj.v1.nii.gz" --flipx
}
export -f process_subject
export DIR PYTHON OUT THREADS DWI_REL BVALS_REL BVECS_REL
find "$HCP_ROOT" -mindepth 1 -maxdepth 1 -type d -name "$SUBJECT_GLOB" -print0 |
    xargs -0 -r -n1 -P "$JOBS" bash -c 'process_subject "$1"' _
"$PYTHON" "$DIR/fit.py" --datadir "$OUT/deformed" --outdir "$OUT/fitted" --ref "$OUT/data/t2.nii.gz" \
    --mask-frac .25 --approx-kappa --debug
for name in alpha beta kappa v1_mu FAav; do cp "$OUT/fitted/$name.nii.gz" "$OUT/data/$name.nii.gz"; done
echo "Atlas: $OUT/data (inspect before use; no work files deleted)"
