#!/bin/bash
#SBATCH --account=def-bashivan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=06:00:00
# Searchlight decoding (loc/ctg/obj, encoding and delay) over DLPFC + IFC for one CNM subject on Trillium (GLMsingle D):
# 8 mm searchlights on the fsLR 32k template midthickness, demean+tuned decoder, 1000-permutation null per vertex.
# usage (from the package root): sbatch --job-name=cnm_sl_<sub> slurm_trillium_searchlight.sh <sub> [n_perm]
# Writes <runs>/glmsingle_wfdelay/64kDense/<sub>/_searchlight/.
set -euo pipefail
SUB=$1; NPERM=${2:-1000}
PKG=${PKG:-$SLURM_SUBMIT_DIR}
REPO=$PKG/code/MULTFSCTRL; CNM=$PKG/out/glm/cnm_root; PY=~/envs/fmri_glm/bin/python
N=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR=$PKG/out/logs/numba_cache
SES=$(ls -d $CNM/behav/$SUB/ses-* | xargs -n1 basename | tr '\n' ' ')
cd $REPO/analysis/glm/misc
echo "== [$(date +%T)] searchlight $SUB n_perm=$NPERM"
$PY searchlight_decoding.py --sub $SUB --sessions $SES --root $CNM --prep_run fmriprep_20.2.6 \
  --dlabel $REPO/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii --surf_dir $PKG/data/templates/fsLR_32k \
  --rois DLPFC IFC --radius 8 --features loc ctg obj --phases Encoding Delay --n_perm $NPERM --n_jobs $N --check 2
echo "== [$(date +%T)] done"
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State
