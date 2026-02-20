#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/ctrl_run"
DERIV_DIR="${BASE_DIR}/derivatives"
SUB="sub-01"

mkdir -p "${BASE_DIR}/confounds/${SUB}"

for session_number in {1..5}; do
  session_id=$(printf "%02d" "$session_number")
  session_dir="${DERIV_DIR}/${SUB}/ses-${session_id}/func"
  conf_sess_dir="${BASE_DIR}/confounds/${SUB}/ses-${session_id}"
  mkdir -p "${conf_sess_dir}"

  # Build matches safely
  shopt -s nullglob
  matches=( "${session_dir}/${SUB}_ses-${session_id}_task-"*_desc-confounds* )
  shopt -u nullglob

  if ((${#matches[@]})); then
    cp -t "${conf_sess_dir}" "${matches[@]}"
  else
    echo "No confounds files found for ${SUB} ses-${session_id} in ${session_dir}"
  fi
done