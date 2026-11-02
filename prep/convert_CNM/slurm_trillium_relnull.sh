#!/bin/bash
#SBATCH --account=def-bashivan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=02:00:00
# Split-half reliability with a permutation null (pattern_reliability.py --n_perm) for CNM subjects on Trillium,
# GLMsingle D, both condition keys, the main ROI set and extra ROI groups (~2 min per subject x key x ROI set).
# Writes *_perm<N>.tsv next to the null-free TSVs (main set: _reliability_pattern/; extras: _reliability_pattern/rois-<g>/).
# usage (from the package root): sbatch --job-name=cnm_relnull slurm_trillium_relnull.sh <n_perm> <sub>...
set -uo pipefail
NPERM=$1; shift
PKG=/scratch/lucasg/wm_fmri/cnm_glmsingle
REPO=$PKG/code/MULTFSCTRL; CNM=$PKG/out/glm/cnm_root; PY=~/envs/fmri_glm/bin/python
N=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
cd $REPO/analysis/glm/misc
failed=()
for SUB in "$@"; do
  SES=$(ls -d $CNM/behav/$SUB/ses-* | xargs -n1 basename | tr '\n' ' ')
  R=$CNM/neural/fmriprep_outs/fmriprep_20.2.6/glm_runs/glmsingle_wfdelay/64kDense/$SUB/_reliability_pattern
  for rois in "DLPFC EVC VTC IT" "IFC"; do
    out=$R; [ "$rois" != "DLPFC EVC VTC IT" ] && out=$R/rois-${rois// /-}
    for k in trial stimulus; do
      echo "== [$(date +%T)] $SUB $rois $k"
      $PY pattern_reliability.py --sub $SUB --sessions $SES --glm_type glmsingle --glm_variant wfdelay --root $CNM \
        --prep_run fmriprep_20.2.6 --dlabel $REPO/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii --rois $rois \
        --n_jobs $N --split_mode repeats --cond_key $k --exclude_tasks dms --n_perm $NPERM --out_dir $out \
        | grep -E "SAVED|INFO" || failed+=("$SUB $rois $k")
    done
  done
done
echo "[$(date +%T)] failed: ${#failed[@]} ${failed[*]:-}"
[ ${#failed[@]} -eq 0 ]
