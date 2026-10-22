#!/usr/bin/env bash
set -euo pipefail

#############################################
# GLOBAL CONFIG — CHANGE THESE AS NEEDED
#############################################

SUB="sub-01"
SESSIONS=( "ses-01" "ses-02" "ses-03" "ses-04" )

TR="1.49"

BEHAV_BASE="/mnt/tempdata/lucas/fmri/recordings/TR/behav"
NEURAL_BASE="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run"

# One joint fit per events variant (each in its own tmux window)
EVENTS_TYPES=( "wfdelay" "wofdelay" )

# Model types whose single-trial betas are written (B: library HRF, C: + GLMdenoise, D: + ridge = final)
SAVE_TYPES="B C D"

GLMSINGLE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/glmsingle_fit.py"

# ---- CONDA CONFIG ----
CONDA_BASE="/home/lucas/miniconda3"
CONDA_ENV="fmri_glm"

# GLMsingle is ONE joint fit across sessions (repeats across sessions drive its cross-validation),
# so this launches one job per events variant; give each enough BLAS threads.
CPUS=16

# tmux session name
TMUX_SESSION="glmsingle_${SUB}"

#############################################
# Helper: start tmux session if missing
#############################################
if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${TMUX_SESSION}" -n "launcher"
fi

# thread caps (important for numpy / MKL / OpenBLAS)
THREAD_EXPORTS="export OMP_NUM_THREADS=${CPUS}; export OPENBLAS_NUM_THREADS=${CPUS}; export MKL_NUM_THREADS=${CPUS}; export NUMEXPR_NUM_THREADS=${CPUS};"

#############################################
# Helper: create and run one job
#############################################
spawn_job () {
  local EVENTS_TYPE="$1"

  # Where logs go
  local LOG_DIR="${NEURAL_BASE}/glm_runs/glmsingle_${EVENTS_TYPE}/64kDense/${SUB}/_logs"
  mkdir -p "${LOG_DIR}"

  local JOB_NAME="glmsingle_${EVENTS_TYPE}"
  local LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"
  local JOB_SCRIPT="${LOG_DIR}/${JOB_NAME}.sh"

  # Write a per-job script file to avoid tmux quoting issues
  cat > "${JOB_SCRIPT}" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
${THREAD_EXPORTS}

echo "[START] ${JOB_NAME}  \$(date)"
echo "HOST=\$(hostname)"

# call env python directly (NO conda activate)
PY="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
if [[ ! -x "\${PY}" ]]; then
  echo "ERROR: env python not found/executable: \${PY}"
  exit 12
fi
"\${PY}" -c "import sys, glmsingle; print('PYTHON_EXE=', sys.executable); print('GLMSINGLE=', glmsingle.__file__)"

"\${PY}" "${GLMSINGLE_SCRIPT}" \\
  --sub "${SUB}" \\
  --sessions ${SESSIONS[*]} \\
  --neural_base "${NEURAL_BASE}" \\
  --behav_base "${BEHAV_BASE}" \\
  --events_variant "${EVENTS_TYPE}" \\
  --tr "${TR}" \\
  --add_fd \\
  --save_types ${SAVE_TYPES} \\
  --overwrite

echo "[DONE] ${JOB_NAME}  \$(date)"
EOF2

  chmod +x "${JOB_SCRIPT}"

  # Create a tmux window for this job and run it, logging stdout+stderr
  tmux new-window -t "${TMUX_SESSION}" -n "${JOB_NAME}" \
    "bash -lc 'bash \"${JOB_SCRIPT}\" 2>&1 | tee \"${LOG_FILE}\"'"

  echo "Spawned tmux window: ${TMUX_SESSION}:${JOB_NAME} (cpus=${CPUS}, log=${LOG_FILE})"
}

#############################################
# Main loop
#############################################
for EVENTS_TYPE in "${EVENTS_TYPES[@]}"; do
  spawn_job "${EVENTS_TYPE}"
done

echo
echo "All jobs spawned."
echo "Attach with: tmux attach -t ${TMUX_SESSION}"
echo "List windows: tmux list-windows -t ${TMUX_SESSION}"
