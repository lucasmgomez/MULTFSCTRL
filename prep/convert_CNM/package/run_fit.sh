#!/bin/bash
# One CNM subject's GLMsingle fit (types B, C, D), with the settings used for sub-01 on Rorqual.
# usage: bash run_fit.sh <sub> [n_threads] [extra glmsingle_fit.py args, e.g. --dry_run]
# env:   PYTHON (default: python), OUT (default: <package>/out/glm_runs), RAW (default: <package>/out/glmsingle_raw/<sub>)
# Measured on sub-01: ~25 h, peak RAM 188 GB, ~15 cores busy on average. RAW gets ~11 GB and is deleted and
# rewritten by GLMsingle: point it at a dedicated folder.
# SLURM example: sbatch --cpus-per-task=16 --mem=220G --time=2-00:00:00 --wrap "bash run_fit.sh sub-03 16"
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SUB=$1; N=${2:-16}; shift; [ $# -gt 0 ] && shift
export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N FLEXIBLAS_NUM_THREADS=$N NUMBA_NUM_THREADS=$N
PY=${PYTHON:-python}
SES=$(ls -d "$HERE/data/behav/$SUB"/ses-* | xargs -n1 basename | tr '\n' ' ')
echo "[START] $(date) $SUB threads=$N sessions: $SES"
time "$PY" -u "$HERE/code/MULTFSCTRL/analysis/glm/glmsingle_fit.py" --sub "$SUB" --sessions $SES \
  --neural_base "$HERE/data/neural/fmriprep_outs/fmriprep_20.2.6" --behav_base "$HERE/data/behav" \
  --events_variant wfdelay --add_fd --start_time 0.686 --save_types B C D \
  --output_root "${OUT:-$HERE/out/glm_runs}" --raw_dir "${RAW:-$HERE/out/glmsingle_raw/$SUB}" "$@"
echo "[DONE] $(date)"
