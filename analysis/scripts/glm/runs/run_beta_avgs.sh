# Define paths to match your previous setup
DATA_ROOT="/mnt/tempdata/lucas/fmri/recordings/TR"
OUTPUT_DIR="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lsa_wfdelay/64kDense/sub-01/_bavgs"

# Create output dir
mkdir -p "$OUTPUT_DIR"

python ../misc/beta_averages.py \
    --subj "sub-01" \
    --sessions ses-01 ses-02 ses-03 ses-04 \
    --behav_dir "${DATA_ROOT}/behav" \
    --betas_dir "${DATA_ROOT}/neural/fmriprep_outs/first_run/glm_runs/lsa_wfdelay/64kDense" \
    --dlabel_path "/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii" \
    --save_path "$OUTPUT_DIR" \
    --events_type "wfdelay" \
    --lateralize "LR" \
    --rois "['SFL', 'i6-8', 's6-8', 'IFJa', 'IFJp', 'IFSp', 'IFSa', '8BM', '8Av', '8Ad', '8BL', '8C', '9m', '9p', '9a', '9-46d', 'a9-46v', 'p9-46v', '46', '44', '45', '47l', '47m', '47s', 'a47r', 'p47r', '10r', '10d', '10v', 'a10p', 'p10p', '10pp', '11l', '13l']"