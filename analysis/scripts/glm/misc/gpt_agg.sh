python gpt_agg.py \
  --data_cache_dir "/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/preprocessed_data" \
  --decode_results_dir "/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay" \
  --dlabel_path "/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii" \
  --save_path "/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay" \
  --subj "sub-01" \
  --lateralize "LR" \
  --rois "['10pp','10v','47s','46','9-46d']" \
  --verify_against_npz