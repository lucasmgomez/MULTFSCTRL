#!/bin/bash
#SBATCH --account=def-bashivan_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
# Phase 2: build the CNM tree (copies neural files, writes TR-format block logs + mapping.csv).
set -euo pipefail
REPO=/home/lucasg/projects/MULTFSCTRL
CNM=/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM
~/envs/fmri_glm/bin/python $REPO/prep/convert_CNM/build_tree.py --scans_root $CNM/sourcedata/scans --root $CNM \
  --mapping_copy $REPO/prep/convert_CNM/mapping.csv "$@"
