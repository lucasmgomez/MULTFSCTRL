BASE=/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/ctrl_run

# Run this first to create necessary directories
mkdir -p \
  "$BASE/derivatives" \
  "$BASE/work" \
  "$BASE/tmp" \
  "$BASE/home" \
  "$BASE/.cache/templateflow"

# Make everything writable by you (and usable by Docker)
chmod -R u+rwX \
  "$BASE/derivatives" \
  "$BASE/work" \
  "$BASE/tmp" \
  "$BASE/home" \
  "$BASE/.cache"

# Do a docker test 
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -v "$BASE/derivatives:/out" \
  alpine sh -lc 'touch /out/_write_test && ls -l /out/_write_test && rm /out/_write_test'

# Make sure temphost is made and writable
TFHOST=/mnt/tempdata/lucas/fmri/.cache/templateflow
mkdir -p "$TFHOST"
chmod -R u+rwX /mnt/tempdata/lucas/fmri/.cache

# Run fmriprep
docker run -ti --rm \
  --user "$(id -u):$(id -g)" \
  --shm-size=64g \
  -e HOME=/home/fmriprep \
  -e XDG_CACHE_HOME=/home/fmriprep/.cache \
  -e TEMPLATEFLOW_HOME=/templateflow \
  -e TMPDIR=/tmp \
  -v "$BASE/bids:/data:ro" \
  -v "$BASE/derivatives:/out" \
  -v "$BASE/work:/work" \
  -v "$BASE/tmp:/tmp" \
  -v "$BASE/home:/home/fmriprep" \
  -v /home/lucas/projects/MULTFSCTRL/prep/fmriprep/licenses:/fs:ro \
  -v "$TFHOST:/templateflow" \
  nipreps/fmriprep:23.2.0 \
  /data /out participant \
  --participant-label 01 \
  --fs-license-file /fs/license.txt \
  -w /work \
  --nthreads 86 --omp-nthreads 8 --mem-mb 300000 \
  --output-spaces fsLR \
  --cifti-output 91k \
  --write-graph \
  --notrack 2>&1 | tee "$BASE/fmriprep_run_$(date +%Y%m%d_%H%M).log"



#   wb_command -cifti-parcellate \
#     sub-01_task-XXX_space-fsLR_den-91k_bold.dtseries.nii \
#     HCP-MMP1.0_32k.dlabel.nii COLUMN \
#     sub-01_task-XXX_space-fsLR_den-91k_desc-Glasser.ptseries.nii

