#!/usr/bin/env bash
set -euo pipefail

#############################################
# GLOBAL CONFIG — EDIT THESE
#############################################

# Subject / session / task identifiers
SUB="sub-01"
SES="ses-03"
TASK="interdms"
ACQ="obj_ABAB"    
RUN="01"

# ROI selection (comma-separated, exact Glasser labels)
ROI_NAMES="L_46_ROI, R_46_ROI"

# Base directories
NEURAL_BASE="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run"
GLASSER_DLABEL="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii"

# Script location
QC_SCRIPT="./qc_lss_betaseries_save.py"

#############################################
# DERIVED PATHS (usually no need to edit)
#############################################

LSS_BETA_DIR="${NEURAL_BASE}/glm_runs/lss/64kDense/${SUB}/${SES}/task-${TASK}_acq-${ACQ}_run-${RUN}"

FIG_DIR="${LSS_BETA_DIR}/figs"

OUT_PREFIX="qc_${ROI_NAMES//,/+}"

#############################################
# SANITY CHECKS
#############################################

echo "Running LSS ROI QC with:"
echo "  Subject:     ${SUB}"
echo "  Session:     ${SES}"
echo "  Task/Run:    ${TASK} ${ACQ} run-${RUN}"
echo "  ROI(s):      ${ROI_NAMES}"
echo "  LSS dir:     ${LSS_BETA_DIR}"
echo "  Dlabel:      ${GLASSER_DLABEL}"
echo "  Figures dir: ${FIG_DIR}"
echo

if [[ ! -d "${LSS_BETA_DIR}" ]]; then
  echo "ERROR: LSS beta directory not found: ${LSS_BETA_DIR}" >&2
  exit 1
fi

if [[ ! -f "${GLASSER_DLABEL}" ]]; then
  echo "ERROR: Glasser dlabel not found: ${GLASSER_DLABEL}" >&2
  exit 1
fi

mkdir -p "${FIG_DIR}"

#############################################
# RUN QC / VISUALIZATION
#############################################

python "${QC_SCRIPT}" \
  --lss_beta_dir "${LSS_BETA_DIR}" \
  --dlabel "${GLASSER_DLABEL}" \
  --roi_names "${ROI_NAMES}" \
  --fig_dir "${FIG_DIR}" \
  --out_prefix "${OUT_PREFIX}"

echo
echo "LSS ROI QC finished successfully."