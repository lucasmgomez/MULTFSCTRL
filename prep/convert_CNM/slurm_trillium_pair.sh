#!/bin/bash
#SBATCH --account=def-bashivan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=24:00:00
# Two CNM subjects' GLMsingle fits on one Trillium node (192 cores, ~750 GB; 96 threads and ~190 GB peak each),
# split in two stages to fit Trillium's 24 h limit (sub-01's single fit took 24 h 53 min):
#   stage 1: --save_types B C  (library HRF + GLMdenoise PC cross-validation -> pcnum; ~10.5 h on sub-01)
#   stage 2: --save_types D --pcstop -<pcnum from stage 1>  (skips the PC cross-validation; ~16.5 h)
# Same betas as one B C D fit: the noise pool and PCs don't depend on pcstop, and B is refit identically.
# usage (from the package root): sbatch --job-name=cnm_s1 slurm_trillium_pair.sh 1 sub-02 sub-03
#        sbatch --dependency=afterok:<stage-1 job> slurm_trillium_pair.sh 2 sub-02 sub-03
# env: PKG (package root, default /scratch/lucasg/wm_fmri/cnm_glmsingle), PYTHON (default ~/envs/fmri_glm/bin/python)
set -euo pipefail
STAGE=$1; shift
PKG=${PKG:-/scratch/lucasg/wm_fmri/cnm_glmsingle}
export PYTHON=${PYTHON:-$HOME/envs/fmri_glm/bin/python} PYTHONNOUSERSITE=1
# $HOME (and so the venv) is read-only on Trillium compute nodes; numba (fracridge) must cache elsewhere
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-${SCRATCH:-/scratch/$USER}/.numba_cache}
mkdir -p "$PKG/out/logs"
pids=()
for SUB in "$@"; do
  if [ "$STAGE" = 1 ]; then
    ARGS=(--save_types B C)
  else
    SUMMARY=$PKG/out/glm_runs/glmsingleB_wfdelay/64kDense/$SUB/_glmsingle/glmsingle_fit_summary.json
    PCNUM=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['pcnum'])" "$SUMMARY")
    ARGS=(--save_types D --pcstop "-$PCNUM")
  fi
  echo "$SUB stage $STAGE: ${ARGS[*]}"
  RAW=$PKG/out/glmsingle_raw/${SUB}_stage$STAGE bash "$PKG/run_fit.sh" "$SUB" 96 "${ARGS[@]}" \
    > "$PKG/out/logs/${SUB}_stage${STAGE}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
