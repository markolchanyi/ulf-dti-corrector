#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

build() {
    "${CXX:-g++}" -O3 -std=c++11 -shared -fPIC -fopenmp "$DIR/cubic_pull.cpp" -o "$DIR/cubic_pull.so"
}

case "$MODE" in
    build) build; exit ;;
    test) build; exec "$PYTHON" "$DIR/test.py" "$@" ;;
    correct) build; exec "$PYTHON" "$DIR/correct.py" "$@" ;;
    help|--help|-h)
        echo 'bash run.sh all --dwi DTI.nii.gz --bvals bvals --bvecs bvecs --out OUTPUT [--threads 2]'
        echo 'bash run.sh preprocess --dwi DTI.nii.gz --bvals bvals --bvecs bvecs --out OUTPUT'
        echo 'bash run.sh correct --help'
        echo 'Options: --grid 20|30|40 (both models; default 30), --fa-tissue-split (default off)'
        echo 'Volumes are selected from b-values; --b0-threshold 50 sets the low-B cutoff (s/mm^2).'
        echo 'bash run.sh test'
        exit ;;
    all|preprocess) ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac

DWI=""
BVALS=""
BVECS=""
OUT=""
ATLAS="$DIR/atlas/data"
THREADS=2
B0_THRESHOLD=50
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dwi) DWI="$2"; shift 2 ;;
        --bvals) BVALS="$2"; shift 2 ;;
        --bvecs) BVECS="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --atlas) ATLAS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --b0-threshold) B0_THRESHOLD="$2"; shift 2 ;;
        # chosen grid spacing for distortion and bias models. We found 30mm to be best for our data, but your distortions (that you care about) may be finer or coarser!!
        --grid) EXTRA+=(--grid "$2"); shift 2 ;;
        # use tissue class separation if your FA is pretty homogenous
        --fa-tissue-split) EXTRA+=(--fa-tissue-split); shift ;;
        --) shift; EXTRA+=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

SPLIT=0
for ((i=0;i<${#EXTRA[@]};i++)); do
    arg="${EXTRA[$i]}"
    if [[ "$arg" == --fa-tissue-split ]]; then SPLIT=1; fi
    if [[ "$arg" == --b0-threshold ]]; then B0_THRESHOLD="${EXTRA[$((i+1))]}"; fi
    if [[ "$arg" == --b0-threshold=* ]]; then B0_THRESHOLD="${arg#*=}"; fi
done
if [[ "$SPLIT" == 1 ]]; then
    for tissue in gm wm; do
        for field in alpha beta mask; do
            [[ -s "$ATLAS/${field}_${tissue}.nii.gz" ]] || { echo "Missing tissue prior: $ATLAS/${field}_${tissue}.nii.gz" >&2; exit 2; }
        done
    done
fi

[[ -n "$DWI" && -n "$BVALS" && -n "$BVECS" && -n "$OUT" ]] || { echo 'Missing input arguments' >&2; exit 2; }
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid thread count' >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
for file in "$DWI" "$BVALS" "$BVECS" "$ATLAS/t2.nii.gz" "$ATLAS/mask.nii.gz" \
    "$ATLAS/alpha.nii.gz" "$ATLAS/beta.nii.gz" "$ATLAS/kappa.nii.gz" "$ATLAS/v1_mu.nii.gz" "$ATLAS/seg.nii.gz"; do
    [[ -s "$file" ]] || { echo "Missing file: $file" >&2; exit 2; }
done
for exe in mrconvert mrcat mrmath dwidenoise mri_synthsr mri_synthstrip mri_super_synth mri_convert \
    antsRegistration antsApplyTransforms ConvertTransformFile N4BiasFieldCorrection; do
    command -v "$exe" >/dev/null || { echo "Missing command: $exe" >&2; exit 2; }
done
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$THREADS" ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
layout=$("$PYTHON" "$DIR/prepare.py" check --dwi "$DWI" --bvals "$BVALS" --bvecs "$BVECS" --b0-threshold "$B0_THRESHOLD" --layout)
read -r NV REF LOW <<< "$layout"
IFS=',' read -r -a low_indices <<< "$LOW"
mkdir -p "$OUT/tmp" "$OUT/preproc" "$OUT/atlas"
OUT="$(cd "$OUT" && pwd)"
ATLAS="$(cd "$ATLAS" && pwd)"
export MPLCONFIGDIR="$OUT/tmp/matplotlib"
TMP="$OUT/tmp"
PRE="$OUT/preproc"
REG="$OUT/atlas"
exec > >(tee "$OUT/run.log") 2>&1

# SynthSR and stripping 
for ((i=0;i<NV;i++)); do
    mrconvert "$DWI" "$TMP/raw_${i}.nii.gz" -coord 3 "$i" -axes 0,1,2 -quiet
done
for ((i=0;i<NV;i++)); do
    mri_synthsr --i "$TMP/raw_${i}.nii.gz" --o "$TMP/synth_${i}.nii.gz" --lowfield --cpu --threads "$THREADS"
    mri_synthstrip -i "$TMP/synth_${i}.nii.gz" -o "$TMP/guide_${i}.nii.gz" --no-csf
    mri_convert "$TMP/guide_${i}.nii.gz" "$TMP/guide_grid_${i}.nii.gz" -rl "$TMP/raw_${REF}.nii.gz" -odt float
done
mrconvert "$TMP/raw_${REF}.nii.gz" "$TMP/aligned_${REF}.nii.gz" -datatype float32 -quiet
for ((i=0;i<NV;i++)); do
    if [[ "$i" == "$REF" ]]; then continue; fi
    antsRegistration -d 3 --verbose 0 --output "[$TMP/vol_${i}_,$TMP/guide_warped_${i}.nii.gz]" \
        --interpolation Linear --initial-moving-transform "[$TMP/guide_grid_${REF}.nii.gz,$TMP/guide_grid_${i}.nii.gz,1]" \
        --transform 'Rigid[0.1]' --metric "MI[$TMP/guide_grid_${REF}.nii.gz,$TMP/guide_grid_${i}.nii.gz,1,32,Regular,0.25]" \
        --convergence '[500x250x0,1e-6,10]' --shrink-factors 8x4x2 --smoothing-sigmas 3x2x1vox \
        --transform 'Affine[0.1]' --metric "MI[$TMP/guide_grid_${REF}.nii.gz,$TMP/guide_grid_${i}.nii.gz,1,32,Regular,0.25]" \
        --convergence '[500x250x0,1e-6,10]' --shrink-factors 4x2x1 --smoothing-sigmas 3x2x1vox
    antsApplyTransforms -d 3 -i "$TMP/raw_${i}.nii.gz" -r "$TMP/guide_grid_${REF}.nii.gz" \
        -o "$TMP/aligned_${i}.nii.gz" -n BSpline -t "$TMP/vol_${i}_0GenericAffine.mat"
    ConvertTransformFile 3 "$TMP/vol_${i}_0GenericAffine.mat" "$TMP/vol_${i}.txt"
done
images=()
for ((i=0;i<NV;i++)); do images+=("$TMP/aligned_${i}.nii.gz"); done
mrcat "${images[@]}" "$PRE/full_sequence.affine.nii.gz" -axis 3 -quiet
if [[ "${#low_indices[@]}" == 1 ]]; then
    mrconvert "$TMP/aligned_${REF}.nii.gz" "$PRE/mean_b0.nii.gz" -quiet
else
    images=()
    for i in "${low_indices[@]}"; do images+=("$TMP/aligned_${i}.nii.gz"); done
    mrcat "${images[@]}" "$TMP/b0s.nii.gz" -axis 3 -quiet
    mrmath "$TMP/b0s.nii.gz" mean "$PRE/mean_b0.nii.gz" -axis 3 -quiet
fi
"$PYTHON" "$DIR/prepare.py" rotate --bvals "$BVALS" --bvecs "$BVECS" --b0-threshold "$B0_THRESHOLD" \
    --affines "$TMP/vol_{i}.txt" --out "$PRE/bvecs.txt"
cp "$BVALS" "$PRE/bvals.txt"
dwidenoise "$PRE/full_sequence.affine.nii.gz" "$PRE/dwi.nii.gz" -nthreads "$THREADS"
mri_synthstrip -i "$PRE/mean_b0.nii.gz" -o "$TMP/mean_b0_stripped.nii.gz" -m "$PRE/mask.nii.gz" --no-csf
mri_super_synth --i "$TMP/mean_b0_stripped.nii.gz" --o "$TMP/supersynth" --mode invivo --device cpu --threads "$THREADS"
mri_convert "$TMP/supersynth/segmentation.mgz" "$PRE/seg.nii.gz" -rl "$PRE/mean_b0.nii.gz" -rt nearest -odt float
mri_convert "$TMP/supersynth/SynthT2.mgz" "$TMP/synth_t2.nii.gz" -rl "$PRE/mean_b0.nii.gz" -rt cubic -odt float
"$PYTHON" "$DIR/prepare.py" mask --input "$TMP/synth_t2.nii.gz" --mask "$PRE/mask.nii.gz" --out "$TMP/synth_t2_masked.nii.gz"

# N4 used only for atlas reg here....
"$PYTHON" "$DIR/prepare.py" mask --input "$PRE/mean_b0.nii.gz" --mask "$PRE/mask.nii.gz" --out "$TMP/mean_b0_masked.nii.gz"
N4BiasFieldCorrection -d 3 -i "$TMP/mean_b0_masked.nii.gz" -x "$PRE/mask.nii.gz" -s 2 \
    -b '[100]' -c '[50x50x30x20,1e-7]' -o "$TMP/b0_n4.nii.gz"
"$PYTHON" "$DIR/prepare.py" mask --input "$TMP/b0_n4.nii.gz" --mask "$PRE/mask.nii.gz" --out "$TMP/fixed.nii.gz"
FIXED="$TMP/synth_t2_masked.nii.gz"
# Match the successful mask-guided pose registration before SyN.
"$PYTHON" "$DIR/prepare.py" pose --mask "$PRE/mask.nii.gz" --atlas-mask "$ATLAS/mask.nii.gz" --out "$TMP/pose_inputs"
antsRegistration -d 3 --float 0 --random-seed 20260927 --output "[$TMP/pose_,$TMP/atlas_t2_pose.nii.gz]" \
    --interpolation Linear --winsorize-image-intensities '[0.005,0.995]' --use-histogram-matching 0 \
    --initial-moving-transform "$TMP/pose_inputs/mask_pose_initial.mat" \
    --masks '[NULL,NULL]' --transform 'Rigid[0.02]' \
    --metric "MI[$FIXED,$ATLAS/t2.nii.gz,0.5,32,Regular,0.5]" \
    --metric "MeanSquares[$TMP/pose_inputs/subject_distance.nii.gz,$TMP/pose_inputs/atlas_distance.nii.gz,0.5,0]" \
    --convergence '[500x300x150,1e-7,20]' --shrink-factors 4x2x1 --smoothing-sigmas 2x1x0vox \
    --masks '[NULL,NULL]' --transform 'Affine[0.02]' \
    --metric "MI[$FIXED,$ATLAS/t2.nii.gz,0.5,32,Regular,0.5]" \
    --metric "MeanSquares[$TMP/pose_inputs/subject_distance.nii.gz,$TMP/pose_inputs/atlas_distance.nii.gz,0.5,0]" \
    --convergence '[500x300x150,1e-7,20]' --shrink-factors 4x2x1 --smoothing-sigmas 2x1x0vox
antsApplyTransforms -d 3 -i "$ATLAS/mask.nii.gz" -r "$FIXED" -o "$TMP/atlas_mask_pose.nii.gz" \
    -n NearestNeighbor -t "$TMP/pose_0GenericAffine.mat"
"$PYTHON" "$DIR/prepare.py" overlap --mask "$PRE/mask.nii.gz" --atlas-mask "$TMP/atlas_mask_pose.nii.gz"
antsRegistration -d 3 --float 0 --random-seed 20260927 --output "[$TMP/base_,$TMP/atlas_t2_base.nii.gz]" \
    --interpolation 'BSpline[3]' --winsorize-image-intensities '[0.005,0.995]' --use-histogram-matching 1 \
    --initial-moving-transform "$TMP/pose_0GenericAffine.mat" --masks "[$PRE/mask.nii.gz,$ATLAS/mask.nii.gz]" \
    --transform 'SyN[0.05,3,1]' --metric "CC[$FIXED,$ATLAS/t2.nii.gz,1,2]" \
    --convergence '[100x70x40,1e-7,15]' --shrink-factors 4x2x1 --smoothing-sigmas 2x1x0vox
antsApplyTransforms -d 3 -i "$ATLAS/seg.nii.gz" -r "$FIXED" -o "$TMP/atlas_seg_base.nii.gz" \
    -n GenericLabel -t "$TMP/base_1Warp.nii.gz" -t "$TMP/base_0GenericAffine.mat" --float 0
antsApplyTransforms -d 3 -i "$ATLAS/mask.nii.gz" -r "$FIXED" -o "$TMP/atlas_mask_base.nii.gz" \
    -n NearestNeighbor -t "$TMP/base_1Warp.nii.gz" -t "$TMP/base_0GenericAffine.mat" --float 0
"$PYTHON" "$DIR/prepare.py" channels --seg "$PRE/seg.nii.gz" --out "$TMP/target_maps"
"$PYTHON" "$DIR/prepare.py" channels --seg "$TMP/atlas_seg_base.nii.gz" --out "$TMP/atlas_maps"

# uses smooth anatomical distance maps for the label guidence. worked nice before
metrics=()
names=(cerebral_wm ventricles thalami deep_gm cc_proxy)
weights=(0.05 0.12 0.14 0.14 0.10)
for i in {0..4}; do
    name="${names[$i]}"
    metrics+=(--metric "CC[$TMP/target_maps/$name.distance.nii.gz,$TMP/atlas_maps/$name.distance.nii.gz,${weights[$i]},2]")
done
antsRegistration -d 3 --float 0 --random-seed 42 --output "[$TMP/refine_,$REG/t2.nii.gz]" \
    --interpolation 'BSpline[3]' --winsorize-image-intensities '[0.005,0.995]' --use-histogram-matching 1 \
    --masks "[$PRE/mask.nii.gz,$TMP/atlas_mask_base.nii.gz]" \
    --transform 'Rigid[0.01]' --metric "MI[$TMP/synth_t2_masked.nii.gz,$TMP/atlas_t2_base.nii.gz,1,32,Regular,0.40]" \
    --convergence '[60x30x15,1e-7,10]' --shrink-factors 4x2x1 --smoothing-sigmas 2x1x0vox \
    --transform 'SyN[0.025,2,1.5]' --metric "CC[$TMP/synth_t2_masked.nii.gz,$TMP/atlas_t2_base.nii.gz,0.45,4]" \
    "${metrics[@]}" --convergence '[140x100x60,1e-7,12]' --shrink-factors 4x2x1 --smoothing-sigmas 2x1x0vox
transforms=(-t "$TMP/refine_1Warp.nii.gz" -t "$TMP/refine_0GenericAffine.mat" -t "$TMP/base_1Warp.nii.gz" -t "$TMP/base_0GenericAffine.mat")
for name in alpha beta kappa; do
    antsApplyTransforms -d 3 -i "$ATLAS/$name.nii.gz" -r "$FIXED" -o "$REG/${name}_deformed_resampled.nii.gz" \
        -n Linear "${transforms[@]}" --float 0
done
if [[ "$SPLIT" == 1 ]]; then
    for tissue in gm wm; do
        for field in alpha beta mask; do
            name="${field}_${tissue}"
            antsApplyTransforms -d 3 -i "$ATLAS/$name.nii.gz" -r "$FIXED" -o "$REG/${name}_deformed_resampled.nii.gz" \
                -n Linear "${transforms[@]}" --float 0
        done
    done
fi
antsApplyTransforms -d 3 -i "$ATLAS/mask.nii.gz" -r "$FIXED" -o "$REG/deformed_mni_mask.nii.gz" \
    -n NearestNeighbor "${transforms[@]}" --float 0
for i in {0..2}; do
    mrconvert "$ATLAS/v1_mu.nii.gz" "$TMP/v1_${i}.nii.gz" -coord 3 "$i" -axes 0,1,2 -quiet
    antsApplyTransforms -d 3 -i "$TMP/v1_${i}.nii.gz" -r "$FIXED" -o "$TMP/v1_warped_${i}.nii.gz" \
        -n Linear "${transforms[@]}" --float 0
done
mrcat "$TMP/v1_warped_0.nii.gz" "$TMP/v1_warped_1.nii.gz" "$TMP/v1_warped_2.nii.gz" "$TMP/v1_warped.nii.gz" -axis 3 -quiet
antsApplyTransforms -d 3 -r "$FIXED" -o "DisplacementField[$REG/pull_lps_mm.nii.gz]" "${transforms[@]}" --float 0
"$PYTHON" "$DIR/prepare.py" vectors --v1 "$TMP/v1_warped.nii.gz" --pull "$REG/pull_lps_mm.nii.gz" \
    --mask "$REG/deformed_mni_mask.nii.gz" --out "$REG/v1mu_deformed_resampled.nii.gz"
"$PYTHON" "$DIR/prepare.py" qc --fixed "$TMP/synth_t2_masked.nii.gz" --moving "$REG/t2.nii.gz" \
    --mask "$PRE/mask.nii.gz" --atlas-mask "$REG/deformed_mni_mask.nii.gz" --out "$REG"
touch "$PRE/COMPLETE"

if [[ "$MODE" == all ]]; then
    build
    "$PYTHON" "$DIR/correct.py" --dwi "$PRE/dwi.nii.gz" --bvals "$PRE/bvals.txt" --bvecs "$PRE/bvecs.txt" \
        --mask "$PRE/mask.nii.gz" --seg "$PRE/seg.nii.gz" --atlas "$REG" --out "$OUT/correction" \
        --threads "$THREADS" --b0-threshold "$B0_THRESHOLD" "${EXTRA[@]}"
fi
