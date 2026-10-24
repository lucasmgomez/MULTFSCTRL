#!/bin/bash
#SBATCH --account=def-bashivan_cpu
# Phase 4 evaluation of one CNM GLM (resources on the sbatch command line).
# usage: sbatch [resources] slurm_evaluate.sh <sub> <glm_type> <variant> <step> [n_perm]
#   step: leakage | reliability | sweep-<feature> (loc, ctg or obj) | plots
# Single-trial betas (--repeat_mode single), dms left out of decoding (HANDOFF.md decisions #5, #6).
# Outputs go to <glm>/64kDense/<sub>/_decoding_roi and _reliability_pattern (the scripts' defaults).
# Env overrides (smoke tests): CNM_ROOT (dataset root), SESSIONS (space-separated; default all of the subject's).
set -euo pipefail
SUB=$1; GT=$2; VAR=$3; STEP=$4; NPERM=${5:-1000}
N=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1
REPO=/home/lucasg/projects/MULTFSCTRL
CNM=${CNM_ROOT:-/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM}
PY=~/envs/fmri_glm/bin/python
DL=$REPO/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii
SES=${SESSIONS:-$(ls -d $CNM/behav/$SUB/ses-* | xargs -n1 basename | tr '\n' ' ')}
C="--sub $SUB --sessions $SES --glm_type $GT --glm_variant $VAR --root $CNM --prep_run fmriprep_20.2.6 --dlabel $DL"
OUT=$CNM/neural/fmriprep_outs/fmriprep_20.2.6/glm_runs/${GT}_${VAR}/64kDense/$SUB/_decoding_roi
cd $REPO/analysis/glm/misc
echo "[START] $(date) $STEP cpus=$N"
case $STEP in
  leakage)
    for f in loc obj; do
      $PY temporal_leakage.py $C --feature $f --rois EVC VTC IT DLPFC --n_perm $NPERM --n_jobs $N \
        --repeat_mode single --exclude_tasks dms
      $PY temporal_leakage.py $C --feature $f --rois EVC VTC --n_perm $NPERM --n_jobs $N \
        --repeat_mode single --tasks 1back
    done ;;
  reliability)
    for k in trial stimulus; do
      $PY pattern_reliability.py $C --rois DLPFC EVC VTC IT --n_jobs $N --split_mode repeats --cond_key $k \
        --exclude_tasks dms
    done ;;
  sweep-*)
    $PY feature_roi_sweep.py $C --features ${STEP#sweep-} --rois DLPFC EVC VTC IT --levels group parcel \
      --configs tuned demean_pos+tuned demean+tuned --n_perm $NPERM --n_jobs $N \
      --repeat_mode single --exclude_tasks dms ;;
  plots)
    T=$(ls $OUT/roi_decoding_${GT}_${VAR}_{loc,ctg,obj}_excl-dms_rep-single_group-parcels-DLPFC-EVC-VTC-IT_cfg-tuned-demean_postuned-demeantuned.tsv)
    for spec in "demean+tuned 1-taskpos-removed" "demean_pos+tuned 2-pos-removed" "tuned 3-nothing-removed"; do
      set -- $spec
      $PY plot_roi_decoding.py $T --config $1 --level parcel --out $OUT/roi_decoding_${GT}_${VAR}_parcels_$2.png \
        --title "CNM $SUB ${GT}_${VAR}: single-trial decoding ($1)"
    done ;;
esac
echo "[DONE] $(date)"
