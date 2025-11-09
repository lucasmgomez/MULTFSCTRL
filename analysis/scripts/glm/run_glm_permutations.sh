#!/bin/bash
#SBATCH --account=def-bashivan
#SBATCH --job-name=glm_all
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=xiaoxuan.lei.claire@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Threading (avoid oversubscription)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Arrays
subjects=("sub-05" "sub-06")
tasks=("1back" "ctxdm" "dms" "interdms")
include_types=("delay" "encoding" "encoding delay")
output_roots=("delay" "encoding" "encoding_delay")

# Env
module purge
module load python/3.10
module load scipy-stack   # Only if you truly need it in addition to your venv
source /project/def-pbellec/xuan/nilearn/bin/activate

cd /project/def-pbellec/xuan/fmri_dataset_project/scripts

echo "Starting GLM analysis permutations..."
echo "Subjects: ${subjects[*]}"
echo "Tasks: ${tasks[*]}"
echo "Include types combos: 'delay' | 'encoding' | 'encoding delay'"
echo ""

# Ensure top-level output roots exist
for root in "${output_roots[@]}"; do
  mkdir -p "/project/def-pbellec/xuan/fmri_dataset_project/data/${root}"
done

# Loops
for subj in "${subjects[@]}"; do
  for task in "${tasks[@]}"; do
    for i in "${!include_types[@]}"; do
      include_type="${include_types[$i]}"
      output_root="${output_roots[$i]}"

      outdir="/project/def-pbellec/xuan/fmri_dataset_project/data/${output_root}"
      logdir="${outdir}/logs/${subj}/${task}"
      mkdir -p "${logdir}"

      echo "Running: ${subj} | ${task} | include_types: ${include_type}"
      echo "Output root: ${outdir}"
      echo "Logs: ${logdir}"
      echo ""

      # IMPORTANT: don't quote $include_type after the flag so it expands into multiple args if it contains a space
      python glm_analysis.py \
        --subj "${subj}" \
        --tasks "${task}" \
        --include_types ${include_type} \
        --out_root "${outdir}" \
        --overwrite \
        > "${logdir}/${subj}_${task}_$(echo "${include_type// /_}").out" \
        2> "${logdir}/${subj}_${task}_$(echo "${include_type// /_}").err"

      status=$?
      if [ $status -eq 0 ]; then
        echo "✓ Completed: ${subj} - ${task} - ${include_type}"
      else
        echo "✗ Failed: ${subj} - ${task} - ${include_type} (exit ${status})"
      fi
      echo ""
    done
  done
done

echo "All GLM analysis permutations completed!"
