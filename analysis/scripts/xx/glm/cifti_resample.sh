#!/bin/bash

# Set base variables
basedir="/project/def-pbellec/xuan/cneuromod.multfs.fmriprep"
glasser_template="/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii"
subj="sub-06"

# Iterate over sessions
for session_number in {1..16}; do
    session_id=$(printf "%03d" "$session_number")
    datadir="${basedir}/${subj}/ses-${session_id}/func"
    targetdir="/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled/${subj}/ses-${session_id}"

    # Create target directory
    mkdir -p "$targetdir"

    echo "Processing session: ses-${session_id}"

    # Iterate over all matching files
    for file_path in "$datadir"/*_space-fsLR_den-91k_bold.dtseries.nii; do
        # Check if the file exists (in case of no match)
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

echo "All sessions processed."
