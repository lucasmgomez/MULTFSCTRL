#!/bin/bash
#SBATCH --account=def-bashivan_cpu
# Compare beta sets on the same evaluation (group ROIs, single trials, dms excluded). Resources on the command line.
# usage: sbatch [resources] slurm_compare.sh <sub> <glm_type> [n_perm]
#   reads <glm_type>_wfdelay from $CMP_ROOT (default /home/lucasg/scratch/CNM/cmp_root), writes to $CMP_OUT/<glm_type>
set -euo pipefail
SUB=$1; GT=$2; NPERM=${3:-100}
N=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1
REPO=/home/lucasg/projects/MULTFSCTRL; PY=~/envs/fmri_glm/bin/python
ROOT=${CMP_ROOT:-/home/lucasg/scratch/CNM/cmp_root}; OUT=${CMP_OUT:-/home/lucasg/scratch/CNM/cmp_out}/$GT
SES=$(ls -d $ROOT/behav/$SUB/ses-* | xargs -n1 basename | tr '\n' ' ')
C="--sub $SUB --sessions $SES --glm_type $GT --glm_variant wfdelay --root $ROOT --prep_run fmriprep_20.2.6
   --dlabel $REPO/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii --out_dir $OUT --exclude_tasks dms"
cd $REPO/analysis/glm/misc
echo "[START] $(date) $GT"
$PY feature_roi_sweep.py $C --features loc ctg obj --rois DLPFC EVC VTC IT --levels group --configs demean+tuned \
  --n_perm $NPERM --n_jobs $N --repeat_mode single
for k in stimulus trial; do
  $PY pattern_reliability.py $C --rois DLPFC EVC VTC IT --levels group --n_jobs $N --split_mode repeats --cond_key $k
done
$PY temporal_leakage.py $C --feature loc --rois EVC VTC --n_perm $NPERM --n_jobs $N --repeat_mode single
echo "[DONE] $(date)"
