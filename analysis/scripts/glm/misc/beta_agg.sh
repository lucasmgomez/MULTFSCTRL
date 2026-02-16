MODEL_PATH="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed"

# 1. Location of s_betas.npy (from your predict script args)
CACHE_DIR="${MODEL_PATH}/preprocessed_data"

# 2. Location of saved Scalers (regressors folder)
RESULTS_DIR="${MODEL_PATH}/results/frame-only_enc+delay_delay_lsa_wfdelay"

python betas_aggregator.py \
    --behav_dir "/mnt/tempdata/lucas/fmri/recordings/TR/behav" \
    --betas_dir "/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lsa_wfdelay/64kDense" \
    --acts_dir "${MODEL_PATH}/activations" \
    --dlabel_path "/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii" \
    --data_cache_dir "$CACHE_DIR" \
    --decode_results_dir "$RESULTS_DIR" \
    --save_path "$RESULTS_DIR" \
    --subj "sub-01" \
    --sessions ses-01 ses-02 ses-03 ses-04 \
    --events_type "wfdelay" \
    --lateralize "LR" \
    --rois "['10pp','10v','47s','46','9-46d']"