#!/bin/bash

# Set base directories
basedir="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/derivatives"
glasser_template="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii"
subj="sub-01"

# Loop over sessions
for session_number in {1..4}; do
    session_id=$(printf "%02d" "$session_number")

    datadir="${basedir}/${subj}/ses-${session_id}/func"
    targetdir="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glasser_resampled/${subj}/ses-${session_id}"

    mkdir -p "$targetdir"
    echo "Processing session: ses-${session_id}"

    # Loop through dtseries files
    for file_path in "$datadir"/*_space-fsLR_den-91k_bold.dtseries.nii; do
        [ -e "$file_path" ] || continue

        filename=$(basename "$file_path")
        prefix="${filename%_space-fsLR_den-91k_bold.dtseries.nii}"

        output_file="${targetdir}/${prefix}_space-Glasser64k_bold.dtseries.nii"

        echo "→ Resampling $filename"
        wb_command -cifti-resample \
            "$file_path" COLUMN \
            "$glasser_template" COLUMN \
            ADAP_BARY_AREA CUBIC \
            "$output_file"
    done
done