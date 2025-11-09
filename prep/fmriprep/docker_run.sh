docker run -ti --rm \
    -v /mnt/tempdata/lucas/fmri/recordings/TR/neural/final_all_ses/bids:/data:ro \
    -v /mnt/tempdata/lucas/fmri/recordings/TR/neural/final_all_ses/derivatives:/out \
    -v /mnt/tempdata/lucas/fmri/recordings/TR/neural/final_all_ses/work:/work \
    -v /home/lucas/projects/MGH_NACC+MULTFS/MULTFS/prep/fmriprep/licenses:/fs \
    -v $HOME/.cache/templateflow:/opt/templateflow \
    nipreps/fmriprep:23.2.0 \
    /data /out participant \
    --participant-label 01 \
    --fs-license-file /fs/license.txt \
    --nthreads 112 --omp-nthreads 8 --mem-mb 480000 \
    --output-spaces fsLR \
    --cifti-output 91k \
    --write-graph \
    --notrack

#   wb_command -cifti-parcellate \
#     sub-01_task-XXX_space-fsLR_den-91k_bold.dtseries.nii \
#     HCP-MMP1.0_32k.dlabel.nii COLUMN \
#     sub-01_task-XXX_space-fsLR_den-91k_desc-Glasser.ptseries.nii

