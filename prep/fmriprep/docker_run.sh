docker run -ti --rm \
    --user $(id -u):$(id -g) \
    --shm-size=64g \
    -e HOME=/home/fmriprep \
    -e XDG_CACHE_HOME=/home/fmriprep/.cache \
    -e TEMPLATEFLOW_HOME=/opt/templateflow \
    -e TMPDIR=/tmp \
    -v /mnt/tempdata/lucas/fmri/recordings/TR/neural/final_all_ses/bids:/data:ro \
    -v /mnt/tempdata/lucas/fmri/recordings/TR/neural/final_all_ses/derivatives:/out \
    -v /mnt/tempdata/lucas/fmri/recordings/TR/neural/final_all_ses/work:/work \
    -v /mnt/tempdata/lucas/fmri/recordings/TR/neural/final_all_ses/tmp:/tmp \
    -v /mnt/tempdata/lucas/fmri/recordings/TR/neural/final_all_ses/home:/home/fmriprep \
    -v /home/lucas/projects/MULTFSCTRL/prep/fmriprep/licenses:/fs \
    -v /mnt/tempdata/lucas/fmri/recordings/TR/neural/final_all_ses/.cache/templateflow:/opt/templateflow \
    nipreps/fmriprep:23.2.0 \
    /data /out participant \
    --participant-label 01 \
    --fs-license-file /fs/license.txt \
    -w /work \
    --nthreads 112 --omp-nthreads 8 --mem-mb 400000 \
    --output-spaces fsLR \
    --cifti-output 91k \
    --write-graph \
    --notrack



#   wb_command -cifti-parcellate \
#     sub-01_task-XXX_space-fsLR_den-91k_bold.dtseries.nii \
#     HCP-MMP1.0_32k.dlabel.nii COLUMN \
#     sub-01_task-XXX_space-fsLR_den-91k_desc-Glasser.ptseries.nii

