#!/usr/bin/env bash
set -euo pipefail

# --------- EDIT THESE PATHS / PARAMS ----------
PY_SCRIPT="/home/lucas/projects/MULTFSCTRL/analysis/scripts/glm/split_half_fisherz.py"

BASE_DIR="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lss_wofdelay/64kDense"
BEHAV_BASE="/mnt/tempdata/lucas/fmri/recordings/TR/behav"
BLOCKFILES_DIR="/home/lucas/projects/task_stimuli/data/multfs/trevor/blockfiles"
SUB="sub-01"
SESSIONS="ses-01,ses-02,ses-03,ses-04"
DLABEL="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii"
EVENTS_TYPE="lss_wofdelay"

OUT_DIR="${BASE_DIR}/${SUB}/_reliability_roi_wofdelay"
# Optional: filter which task dirs
ONLY_TASKS_REGEX=""   # e.g. "interdms|ctxdm" (leave empty for all)

# Flags
SPEARMAN_BROWN="--spearman_brown"
VERBOSE="--verbose"
# ---------------------------------------------

# Your ROI shortnames
names=(
  SFL i6-8 s6-8 IFJa IFJp IFSp IFSa 8BM 8Av 8Ad 8BL
  8C 9m 9p 9a 9-46d a9-46v p9-46v 46 44 45 47l 47m
  47s a47r p47r 10r 10d 10v a10p p10p 10pp 11l 13l
)

mkdir -p "$OUT_DIR"

for n in "${names[@]}"; do
  ROI="L_${n}_ROI, R_${n}_ROI"
  echo "========================================"
  echo "[RUN] ROI: $ROI"
  echo "========================================"

  cmd=(python3 "$PY_SCRIPT"
    --base_dir "$BASE_DIR"
    --behav_base "$BEHAV_BASE"
    --blockfiles_dir "$BLOCKFILES_DIR"
    --events_type "$EVENTS_TYPE"
    --sub "$SUB"
    --sessions "$SESSIONS"
    --dlabel "$DLABEL"
    --roi_names "$ROI"
    --out_dir "$OUT_DIR"
  )

  # Optional filter
  if [[ -n "$ONLY_TASKS_REGEX" ]]; then
    cmd+=(--only_tasks_regex "$ONLY_TASKS_REGEX")
  fi

  # Optional flags
  if [[ -n "$SPEARMAN_BROWN" ]]; then cmd+=($SPEARMAN_BROWN); fi
  if [[ -n "$VERBOSE" ]]; then cmd+=($VERBOSE); fi

  "${cmd[@]}"
done

echo "Done. Outputs appended in: ${OUT_DIR}/split_half_roi_reliability_fisherz.tsv"