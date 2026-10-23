#!/bin/bash
#SBATCH --account=def-bashivan_cpu
# Resources are set on the sbatch command line (--cpus-per-task, --mem, --time).
# usage: sbatch [resources] slurm_glmsingle.sh <sub> <variant> <output_root> <raw_dir> <sessions...> -- [extra glmsingle_fit args]
# Wall time is printed at the end of the log; peak memory: sacct -j <id> -o MaxRSS,Elapsed.
set -euo pipefail
SUB=$1; VARIANT=$2; OUT=$3; RAW=$4; shift 4
SES=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do SES+=("$1"); shift; done; [ $# -gt 0 ] && shift
N=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N FLEXIBLAS_NUM_THREADS=$N NUMBA_NUM_THREADS=$N
REPO=/home/lucasg/projects/MULTFSCTRL
CNM=/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM
echo "[START] $(date) host=$(hostname) cpus=$N sub=$SUB variant=$VARIANT sessions=${SES[*]} extra=$*"
time ~/envs/fmri_glm/bin/python -u $REPO/analysis/glm/glmsingle_fit.py --sub $SUB --sessions "${SES[@]}" \
  --neural_base $CNM/neural/fmriprep_outs/fmriprep_20.2.6 --behav_base $CNM/behav --events_variant $VARIANT \
  --add_fd --start_time 0.686 --output_root $OUT --raw_dir $RAW "$@"
echo "[DONE] $(date)"
