#!/bin/bash
#SBATCH --account=def-bashivan_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
# Phase 2: score the CNM block logs and build base events (after build_tree.py).
#   events_wfdelay      main variant: 0.5 s pos-1 delays dropped (--min_delay_s 1.0), final delay 3.98 s
#   events_wfdelaypos1  pilot only (decision #3): pos-1 delays kept
# usage: sbatch slurm_make_events.sh [subs...]   (default: all five)
set -euo pipefail
REPO=/home/lucasg/projects/MULTFSCTRL
CNM=/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM
PY=~/envs/fmri_glm/bin/python
SUBS=${@:-sub-01 sub-02 sub-03 sub-05 sub-06}
for sub in $SUBS; do
  for d in $CNM/behav/$sub/ses-*; do
    # raw logs only: the folder glob would also pick up *_scored.tsv on a rerun
    $PY $REPO/prep/behav/score_responses.py "$d/*_events_??-??-??.tsv" > /dev/null
    $PY $REPO/analysis/glm/create_events_lsa.py -i $d -o $d/events_wfdelay --final_delay_duration 3.98 \
      --min_delay_s 1.0 > /dev/null
    if [ $sub = sub-01 ]; then
      $PY $REPO/analysis/glm/create_events_lsa.py -i $d -o $d/events_wfdelaypos1 --final_delay_duration 3.98 > /dev/null
    fi
    echo "$sub $(basename $d): $(ls $d/*_scored.tsv | wc -l) scored, $(ls $d/events_wfdelay/*_base-events.tsv | wc -l) base-events"
  done
done
