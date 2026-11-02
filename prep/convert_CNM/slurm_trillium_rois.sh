#!/bin/bash
#SBATCH --account=def-bashivan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=24:00:00
# Extra ROI groups for one CNM subject on Trillium (GLMsingle D): split-half reliability and the loc/ctg/obj sweeps
# (vertex patterns and ROI means), same settings as slurm_evaluate.sh (single trials, dms excluded, 3 configs).
# usage (from the package root): sbatch --job-name=cnm_rois_<sub> slurm_trillium_rois.sh <sub> <n_perm> <ROI group>...
#   e.g. ... sub-02 1000 IFC      (IFC = ventrolateral PFC: 44, 45, IFJp, IFJa, IFSp, IFSa, 47l, p47r)
# Decoding TSVs go to _decoding_roi (the ROI list is in their name); reliability TSVs to
# _reliability_pattern/rois-<groups>/ (pattern_reliability.py's file name doesn't include the ROIs).
set -uo pipefail
SUB=$1; NPERM=$2; shift 2; ROIS="$*"
PKG=/scratch/lucasg/wm_fmri/cnm_glmsingle
REPO=$PKG/code/MULTFSCTRL; CNM=$PKG/out/glm/cnm_root; PY=~/envs/fmri_glm/bin/python
N=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
SES=$(ls -d $CNM/behav/$SUB/ses-* | xargs -n1 basename | tr '\n' ' ')
C="--sub $SUB --sessions $SES --glm_type glmsingle --glm_variant wfdelay --root $CNM --prep_run fmriprep_20.2.6
   --dlabel $REPO/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii"
G=$CNM/neural/fmriprep_outs/fmriprep_20.2.6/glm_runs/glmsingle_wfdelay/64kDense/$SUB
cd $REPO/analysis/glm/misc
failed=()
step() { echo "== [$(date +%T)] $*"; "$@" || failed+=("$*"); }
for k in trial stimulus; do
  step $PY pattern_reliability.py $C --rois $ROIS --n_jobs $N --split_mode repeats --cond_key $k --exclude_tasks dms \
    --out_dir $G/_reliability_pattern/rois-${ROIS// /-}
done
for f in loc ctg obj; do
  for r in vertex mean; do
    step $PY feature_roi_sweep.py $C --features $f --rois $ROIS --levels group parcel \
      --configs tuned demean_pos+tuned demean+tuned --n_perm $NPERM --n_jobs $N \
      --repeat_mode single --exclude_tasks dms --reduce $r
  done
done
echo "[$(date +%T)] failed steps: ${#failed[@]} ${failed[*]:-}"
[ ${#failed[@]} -eq 0 ]
