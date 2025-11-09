docker run -ti --rm \
    -v /mnt/tempdata/lucas/fmri/recordings/TR/neural/ses-1/bids:/data:ro \
    -v /mnt/tempdata/lucas/fmri/recordings/TR/neural/ses-1/derivatives:/out \
    -v /mnt/tempdata/lucas/fmri/tmp/fmriprep/work:/work \
    -v /home/lucas/projects/MGH_NACC+MULTFS/MULTFS/prep/fmriprep/licenses:/fs \
    -v $HOME/.cache/templateflow:/opt/templateflow \
    nipreps/fmriprep:23.2.0 \
    /data /out participant \
    --participant-label 01 \
    --fs-license-file /fs/license.txt \
    --nthreads 64 --omp-nthreads 16 --mem-mb 400000 \
    --output-spaces fsLR \
    --cifti-output 91k \
    --stop-on-first-crash \
    --write-graph \
    --notrack

#   wb_command -cifti-parcellate \
#     sub-01_task-XXX_space-fsLR_den-91k_bold.dtseries.nii \
#     HCP-MMP1.0_32k.dlabel.nii COLUMN \
#     sub-01_task-XXX_space-fsLR_den-91k_desc-Glasser.ptseries.nii

