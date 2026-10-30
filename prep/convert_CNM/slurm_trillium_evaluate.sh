#!/bin/bash
#SBATCH --account=def-bashivan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
# Full evaluation of one CNM subject on Trillium (whole node; 128 joblib workers ≈ 3 GB each, 1 BLAS thread each):
# slurm_evaluate.sh steps for GLMsingle D (reliability, leakage, sweeps with vertex patterns and ROI means, parcel
# plots), leakage for B, and diag/loc_by_run.py. Every step runs even if an earlier one fails; exit 1 if any failed.
# usage (from the package root): sbatch --job-name=cnm_eval_<sub> slurm_trillium_evaluate.sh <sub> [n_perm]
# env: PKG (package root, default /scratch/lucasg/wm_fmri/cnm_glmsingle), CNM_ROOT (dataset root with glm_runs;
# default <PKG>/out/cnm_root, symlinks to data/ and out/glm_runs), SESSIONS (smoke tests; see slurm_evaluate.sh)
set -uo pipefail
SUB=$1; NPERM=${2:-1000}
PKG=${PKG:-/scratch/lucasg/wm_fmri/cnm_glmsingle}
export REPO=$PKG/code/MULTFSCTRL CNM_ROOT=${CNM_ROOT:-$PKG/out/cnm_root} PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR=${SCRATCH:-/scratch/$USER}/.numba_cache
E=$REPO/prep/convert_CNM/slurm_evaluate.sh
D=$CNM_ROOT/neural/fmriprep_outs/fmriprep_20.2.6/glm_runs/glmsingle_wfdelay/64kDense/$SUB/_decoding_roi
mkdir -p $D
failed=()
step() { echo "== [$(date +%T)] $*"; "$@" || failed+=("$*"); }
step bash $E $SUB glmsingle wfdelay reliability
step bash $E $SUB glmsingle wfdelay leakage $NPERM
step bash $E $SUB glmsingleB wfdelay leakage $NPERM
step env LOC_BY_RUN_OUT=$D/loc_by_run_EVC.tsv OMP_NUM_THREADS=16 \
  ~/envs/fmri_glm/bin/python $REPO/prep/convert_CNM/diag/loc_by_run.py $SUB glmsingle wfdelay EVC
for f in loc ctg obj; do
  step bash $E $SUB glmsingle wfdelay sweep-$f $NPERM
  step env REDUCE=mean bash $E $SUB glmsingle wfdelay sweep-$f $NPERM
done
step bash $E $SUB glmsingle wfdelay plots
echo "[$(date +%T)] failed steps: ${#failed[@]} ${failed[*]:-}"
[ ${#failed[@]} -eq 0 ]
