#!/bin/bash
#SBATCH --account=def-bashivan
#SBATCH --job-name=nilearn_glm_all
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-user=xiaoxuan.lei.claire@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Threading control (avoid oversubscription)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# === Paths ===
SCRIPT_DIR="/project/def-pbellec/xuan/fmri_dataset_project/scripts"
ANALYSIS="${SCRIPT_DIR}/glm_analysis_nilearn.py"
EVAL="${SCRIPT_DIR}/sanity_check/glm_evaluation_nilearn.py"

FMRI_ROOT="/project/def-pbellec/xuan/cneuromod.multfs.fmriprep"
CONF_ROOT="/project/def-pbellec/xuan/cneuromod.multfs.fmriprep"
EVENTS_ROOT="/project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior"
OUT_ROOT="/project/def-pbellec/xuan/fmri_dataset_project/data/nilearn_data"
EVAL_OUT="/project/def-pbellec/xuan/fmri_dataset_project/scripts/sanity_check/glm_fitting_check_results/nilearn_glm"

# === Arrays ===
subjects=("sub-01" "sub-02" "sub-03" "sub-05" "sub-06")
tasks=("1back" "ctxdm" "dms" "interdms")
include_types=("encoding" "delay" "encoding_delay")

# === Conda env ===
module purge
module load python/3.10
module load scipy-stack   # Only if truly needed alongside your venv
source /project/def-pbellec/xuan/nilearn/bin/activate

cd "$SCRIPT_DIR"

echo "=== Starting Nilearn voxel-level GLM analysis ==="
echo "OUT_ROOT: $OUT_ROOT"
echo

# Ensure top-level include-type roots exist
mkdir -p "$OUT_ROOT/encoding" "$OUT_ROOT/delay" "$OUT_ROOT/encoding_delay"

for subj in "${subjects[@]}"; do
  for task in "${tasks[@]}"; do
    for inc in "${include_types[@]}"; do

      echo ">>> Running GLM for $subj | $task | include_types='$inc'"

      # IMPORTANT: don't quote $inc after the flag so it expands into multiple args if it contains a space
      python "$ANALYSIS" \
        --subj "$subj" \
        --tasks "$task" \
        --include_types $inc \
        --fmri_root "$FMRI_ROOT" \
        --conf_root "$CONF_ROOT" \
        --events_root "$EVENTS_ROOT" \
        --out_root "$OUT_ROOT" \
        --overwrite

      echo "✓ GLM completed: $subj $task ($inc)"
      echo ""
    done

    # Run evaluation only on run-01 for speed
    # We show example for 'encoding' include_type. You can loop others similarly if desired.
    echo ">>> Running evaluation for $subj | $task | ses-001 | run-01 (encoding)"
    python "$EVAL" \
      --subj "$subj" \
      --task "$task" \
      --ses ses-001 \
      --run run-01 \
      --tmask 1 \
      --glm_root "$OUT_ROOT/encoding/trial_level_betas" \
      --fmri_root "$FMRI_ROOT" \
      --output_dir "$EVAL_OUT" \
      --max_voxels 3000 \
      --seed 123

    echo "✓ Evaluation completed: $subj $task (run-01)"
    echo ""

  done
done

echo "=== All jobs finished ==="
