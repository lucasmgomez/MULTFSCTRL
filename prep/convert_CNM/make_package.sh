#!/bin/bash
#SBATCH --account=def-bashivan_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
# Build a portable tar with code + inputs to run the CNM GLMsingle fits elsewhere (see package/README.md).
# usage: sbatch make_package.sh <out_dir> [subs...]   (default subs: sub-02 sub-03 sub-05 sub-06)
# Stages symlinks, then tar -h copies the real files. Applies the CLAUDE.md run exclusions (sub-02 ses-10 1back loc).
set -euo pipefail
OUTDIR=$1; shift; SUBS=${@:-sub-02 sub-03 sub-05 sub-06}
REPO=/home/lucasg/projects/MULTFSCTRL
CNM=/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM
GLMS_COMMIT=1ab54a65edd3ea41a6133d4b4ecb78a9c7296684
NAME=cnm_glmsingle
ST=$OUTDIR/stage/$NAME
rm -rf "$OUTDIR/stage"; mkdir -p "$ST/code" "$ST/data/behav" "$ST/data/neural/fmriprep_outs/fmriprep_20.2.6"/{64kDense,confounds}

# code: repo at HEAD (committed state only) + GLMsingle source at the pinned commit
mkdir -p "$ST/code/MULTFSCTRL"; git -C $REPO archive HEAD | tar -x -C "$ST/code/MULTFSCTRL"
git -C $REPO rev-parse HEAD > "$ST/code/COMMIT"
TMPG=$OUTDIR/glmsingle_src; rm -rf "$TMPG"; git clone -q https://github.com/cvnlab/GLMsingle "$TMPG"
mkdir -p "$ST/code/GLMsingle-1ab54a6"; git -C "$TMPG" archive $GLMS_COMMIT | tar -x -C "$ST/code/GLMsingle-1ab54a6"
cp $REPO/prep/convert_CNM/package/{README.md,run_fit.sh} "$ST/"
cat > "$ST/requirements.txt" <<'EOF'
numpy==2.4.2
scipy==1.16.2
pandas==2.3.3
scikit-learn==1.8.0
nilearn==0.12.1
nibabel==5.3.3
joblib==1.5.3
matplotlib==3.10.8
h5py
tqdm
numba
fracridge==2.0
EOF

# data (symlinks; the tar dereferences them). Excluded runs: (sub, ses, behaviour stem, fMRIPrep stem)
EXCL="sub-02 ses-10 task-1back_loc task-1back_acq-loc_run-01"
for sub in $SUBS; do
  for d in $CNM/behav/$sub/ses-*; do
    ses=$(basename $d); dst=$ST/data/behav/$sub/$ses; mkdir -p $dst/events_wfdelay
    ln -s $d/*_events_??-??-??.tsv $d/*_events_??-??-??_scored.tsv $dst/
    ln -s $d/events_wfdelay/*_base-events.tsv $dst/events_wfdelay/
  done
  for k in 64kDense confounds; do ln -s $CNM/neural/fmriprep_outs/fmriprep_20.2.6/$k/$sub $ST/data/neural/fmriprep_outs/fmriprep_20.2.6/$k/$sub; done
done
set -- $EXCL
if [[ " $SUBS " == *" $1 "* ]]; then
  rm -v $ST/data/behav/$1/$2/*${3}_block_0_* $ST/data/behav/$1/$2/events_wfdelay/${3}_run-01_base-events.tsv
  for k in 64kDense confounds; do   # replace the subject folder link by per-file links without the excluded run
    l=$ST/data/neural/fmriprep_outs/fmriprep_20.2.6/$k/$1; rm $l
    for s in $CNM/neural/fmriprep_outs/fmriprep_20.2.6/$k/$1/ses-*; do
      mkdir -p $l/$(basename $s); for f in $s/*; do [[ $f == *"$1_$2_$4_"* ]] || ln -s $f $l/$(basename $s)/; done
    done
  done
fi
~/envs/fmri_glm/bin/python - "$CNM/mapping.csv" "$ST/data/mapping.csv" $SUBS <<'EOF'
import sys, pandas as pd
m = pd.read_csv(sys.argv[1]); m = m[m["sub"].isin(sys.argv[3:])].copy()
x = (m["sub"] == "sub-02") & (m.new_ses == "ses-10") & (m.task == "1back") & (m.acq == "loc") & (m.new_run == 1)
m.loc[x, "include"] = 0
m.loc[x, "notes"] = "dropped 2026-10-26: 2,787 empty vertices (incl. 361 DLPFC), highest motion (see CLAUDE.md)"
m.to_csv(sys.argv[2], index=False)
EOF
(cd "$ST" && find -L data -type f -printf "%P\t%s\n" | sort) > "$ST/FILES.tsv"

# sanity: dry run of every subject from the staged tree
for sub in $SUBS; do
  OUT=$OUTDIR/dry RAW=$OUTDIR/dry/raw PYTHON=~/envs/fmri_glm/bin/python bash "$ST/run_fit.sh" $sub 2 --dry_run 2>&1 \
    | grep -E "^Runs|^Conditions|^Design" | sed "s/^/$sub: /"
done

tar -C "$OUTDIR/stage" -chf "$OUTDIR/$NAME.tar" $NAME
(cd "$OUTDIR" && sha256sum $NAME.tar > $NAME.tar.sha256)
ls -la "$OUTDIR/$NAME.tar"; cat "$OUTDIR/$NAME.tar.sha256"; echo "files in tar: $(tar -tf $OUTDIR/$NAME.tar | wc -l)"
