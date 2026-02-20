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

EVENTS_TYPE="wofdelay"

LSS_SCRIPT="/home/lucas/projects/MULTFSCTRL/analysis/scripts/glm/lss.py"

# ---- CONDA CONFIG ----
CONDA_BASE="/home/lucas/miniconda3"
CONDA_ENV="fmri_glm"

# Where logs go
LOG_DIR="${NEURAL_BASE}/glm_runs/lss_${EVENTS_TYPE}/64kDense/${SUB}/_logs"
mkdir -p "${LOG_DIR}"

# tmux session name
TMUX_SESSION="lss_${SUB}"

#############################################
# TASK DEFINITIONS
#############################################

TASKS=(
    # 1-back
    "1back|obj|obj|01|3"
    "1back|ctg|ctg|01|3"
    "1back|loc|loc|01|3"
    # InterDMS
    "interdms|obj_ABAB|objABAB|01|3"
    "interdms|obj_ABBA|objABBA|01|3"
    "interdms|ctg_ABAB|ctgABAB|01|3"
    "interdms|ctg_ABBA|ctgABBA|01|3"
    "interdms|loc_ABAB|locABAB|01|3"
    "interdms|loc_ABBA|locABBA|01|3"
    # ctxdm
    "ctxdm|col|col|01,02|3"
    "ctxdm|lco|lco|01,02|3"
)

#############################################
# Optional: CPU pinning
#############################################
PIN_CPUS=0
CPU_CURSOR=0
TOTAL_CPUS="$(nproc)"

#############################################
# Helper: start tmux session if missing
#############################################
if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${TMUX_SESSION}" -n "launcher"
fi

#############################################
# Helper: create and run one job
#############################################
spawn_job () {
  local SES="$1"
  local TASK="$2"
  local ACQ_BEHAV="$3"
  local ACQ_FMRIPREP="$4"
  local RUN="$5"
  local CPUS="$6"

  local DT_DIR="${NEURAL_BASE}/64kDense/${SUB}/${SES}"
  local CONF_DIR="${NEURAL_BASE}/confounds/${SUB}/${SES}"

  local DT_FILE
  local CONF_TSV
  local EVENTS_DIR
  local OUT_DIR

  if [[ -n "${ACQ_FMRIPREP}" ]]; then
    DT_FILE="${DT_DIR}/${SUB}_${SES}_task-${TASK}_acq-${ACQ_FMRIPREP}_run-${RUN}_space-Glasser64k_bold.dtseries.nii"
    CONF_TSV="${CONF_DIR}/${SUB}_${SES}_task-${TASK}_acq-${ACQ_FMRIPREP}_run-${RUN}_desc-confounds_timeseries.tsv"
  else
    DT_FILE="${DT_DIR}/${SUB}_${SES}_task-${TASK}_run-${RUN}_space-Glasser64k_bold.dtseries.nii"
    CONF_TSV="${CONF_DIR}/${SUB}_${SES}_task-${TASK}_run-${RUN}_desc-confounds_timeseries.tsv"
  fi

  if [[ -n "${ACQ_BEHAV}" ]]; then
    EVENTS_DIR="${BEHAV_BASE}/${SUB}/${SES}/events_lss_${EVENTS_TYPE}/task-${TASK}_${ACQ_BEHAV}_run-${RUN}_LSS"
    OUT_DIR="${NEURAL_BASE}/glm_runs/lss_${EVENTS_TYPE}/64kDense/${SUB}/${SES}/task-${TASK}_acq-${ACQ_BEHAV}_run-${RUN}"
  else
    EVENTS_DIR="${BEHAV_BASE}/${SUB}/${SES}/events_lss_${EVENTS_TYPE}/task-${TASK}_run-${RUN}_LSS"
    OUT_DIR="${NEURAL_BASE}/glm_runs/lss_${EVENTS_TYPE}/64kDense/${SUB}/${SES}/task-${TASK}_run-${RUN}"
  fi

  local JOB_NAME="${SES}_${TASK}"
  if [[ -n "${ACQ_BEHAV}" ]]; then JOB_NAME+="_${ACQ_BEHAV}"; fi
  JOB_NAME+="_run${RUN}"

  local LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"
  local JOB_SCRIPT="${LOG_DIR}/${JOB_NAME}.sh"

  local PIN_PREFIX=""
  if [[ "${PIN_CPUS}" == "1" ]]; then
    local START="${CPU_CURSOR}"
    local END=$((CPU_CURSOR + CPUS - 1))
    if (( END >= TOTAL_CPUS )); then
      START=0
      END=$((CPUS - 1))
      CPU_CURSOR=$((CPUS))
    else
      CPU_CURSOR=$((END + 1))
    fi
    PIN_PREFIX="taskset -c ${START}-${END} "
  fi

  # thread caps (important for numpy / MKL / OpenBLAS)
  local THREAD_EXPORTS="export OMP_NUM_THREADS=${CPUS}; export OPENBLAS_NUM_THREADS=${CPUS}; export MKL_NUM_THREADS=${CPUS}; export NUMEXPR_NUM_THREADS=${CPUS};"

  # Write a per-job script file to avoid tmux quoting issues
  cat > "${JOB_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
${THREAD_EXPORTS}

echo "[START] ${JOB_NAME}  \$(date)"
echo "HOST=\$(hostname)"

# ---- OPTION B: call env python directly (NO conda activate) ----
PY="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
if [[ ! -x "\${PY}" ]]; then
  echo "ERROR: env python not found/executable: \${PY}"
  echo "Check CONDA_BASE=${CONDA_BASE} and CONDA_ENV=${CONDA_ENV}"
  exit 12
fi

echo "PYTHON=\${PY}"
"\${PY}" -c "import sys; print('PYTHON_EXE=', sys.executable); print('PYTHON_VERSION=', sys.version)"

echo "DT_FILE=${DT_FILE}"
echo "EVENTS_DIR=${EVENTS_DIR}"
echo "CONF_TSV=${CONF_TSV}"
echo "OUT_DIR=${OUT_DIR}"
mkdir -p "${OUT_DIR}"

# sanity checks
[[ -f "${DT_FILE}" ]] || { echo "ERROR missing dtseries: ${DT_FILE}"; exit 2; }
[[ -d "${EVENTS_DIR}" ]] || { echo "ERROR missing events dir: ${EVENTS_DIR}"; exit 3; }
[[ -f "${CONF_TSV}" ]] || { echo "ERROR missing confounds: ${CONF_TSV}"; exit 4; }

${PIN_PREFIX}"\${PY}" "${LSS_SCRIPT}" \
  --dtseries "${DT_FILE}" \
  --lss_events_dir "${EVENTS_DIR}" \
  --confounds_tsv "${CONF_TSV}" \
  --tr "${TR}" \
  --output_dir "${OUT_DIR}" \
  --add_fd \
  --save_dscalar \
  --noise_model "ols" \
  --zscore_time \
  --overwrite

echo "[DONE] ${JOB_NAME}  \$(date)"
EOF

  chmod +x "${JOB_SCRIPT}"

  # Create a tmux window for this job and run it, logging stdout+stderr
  tmux new-window -t "${TMUX_SESSION}" -n "${JOB_NAME}" \
    "bash -lc 'bash \"${JOB_SCRIPT}\" 2>&1 | tee \"${LOG_FILE}\"'"

  echo "Spawned tmux window: ${TMUX_SESSION}:${JOB_NAME} (cpus=${CPUS})"
}

#############################################
# Main loops
#############################################
for SES in "${SESSIONS[@]}"; do
  for entry in "${TASKS[@]}"; do
    IFS="|" read -r TASK ACQ_BEHAV ACQ_FMRIPREP RUNS CPUS <<< "${entry}"

    IFS="," read -ra RUN_LIST <<< "${RUNS}"
    for RUN in "${RUN_LIST[@]}"; do
      spawn_job "${SES}" "${TASK}" "${ACQ_BEHAV}" "${ACQ_FMRIPREP}" "${RUN}" "${CPUS}"
    done
  done
done

echo
echo "All jobs spawned."
echo "Attach with: tmux attach -t ${TMUX_SESSION}"
echo "List windows: tmux list-windows -t ${TMUX_SESSION}"