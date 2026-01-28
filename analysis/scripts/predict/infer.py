import os
import pandas as pd
import re
import numpy as np
import glob
import nibabel as nib
import torch
import json
import argparse
import ast
from joblib import dump
from models.regression import pca_ridge_decode

def get_activations(acts_task_dir, tc):
    acts_dict = torch.load(acts_task_dir)
    all_acts = acts_dict['layer_activations']
    all_tcs = acts_dict['tcs']

    acts = all_acts[:, all_tcs.index(tc)]
    return acts.numpy()

def select_data(betas, acts, phase2predict='encoding'):
    stacked_s_betas = []
    stacked_s_acts = []

    for run in betas.keys():
        for tc in betas[run].keys():
            selected_betas = betas[run][tc]
            if tc not in acts[run]: 
                continue 
            selected_acts = acts[run][tc]

            # Beta processing
            if '1back' in run:
                selected_betas = selected_betas[:10, :]  
            n_betas = selected_betas.shape[0]

            if phase2predict == 'encoding':
                selected_betas = selected_betas[::2, :] 
            elif phase2predict == 'delay':
                selected_betas = selected_betas[1::2, :]  

            n_tokens = selected_acts.shape[1]
            start_idx = max(0, n_tokens - n_betas)
            selected_acts = selected_acts[:, start_idx:] 

            encoding_idxs = [i for i in range(0, selected_acts.shape[1], 2)]
            delay_idxs = [i for i in range(1, selected_acts.shape[1], 2)]
            
            if len(encoding_idxs) == 0 or len(delay_idxs) == 0: continue

            enc_acts = selected_acts[:, encoding_idxs, :]
            delay_acts = selected_acts[:, delay_idxs, :]
            selected_acts = np.concatenate((enc_acts, delay_acts), axis=-1)

            stacked_s_betas.append(selected_betas)
            stacked_s_acts.append(selected_acts)

    if not stacked_s_betas:
        raise ValueError("No data found after selection process.")

    s_betas = np.concatenate(stacked_s_betas, axis=0)
    s_acts = np.concatenate(stacked_s_acts, axis=1)

    return s_betas, s_acts

def predict(betas, acts, model, avg_vertices, standardize_acts, standardize_betas, **model_kwargs):
    result, _, regressor = model(acts, betas, avg_vertices=avg_vertices, standardize_acts=standardize_acts, standardize_betas=standardize_betas, **model_kwargs)
    return result, _, regressor

def main():
    parser = argparse.ArgumentParser(description="fMRI PCA Ridge Decoding")

    # Path arguments
    parser.add_argument("--behav_dir", type=str, default="/mnt/tempdata/lucas/fmri/recordings/TR/behav",
                        help="Path to behavior directory")
    parser.add_argument("--betas_dir", type=str, default="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lss/64kDense",
                        help="Path to betas directory")
    parser.add_argument("--acts_dir", type=str, default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/activations",
                        help="Path to activations directory")
    parser.add_argument("--dlabel_path", type=str, default="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii",
                        help="Path to dlabel nifti file")
    parser.add_argument("--save_path", type=str, default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/test",
                        help="Root path to save results")
    
    # Run configuration
    parser.add_argument("--run_name", type=str, default="test_run", help="Name of the current run/experiment")
    parser.add_argument("--subj", type=str, default="sub-01", help="Subject ID")
    parser.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02", "ses-03", "ses-04"],
                        help="List of sessions (space separated)")
    parser.add_argument("--lateralize", type=str, choices=['LR', 'L', 'R'], default='LR',
                        help="Lateralization of ROIs")
    parser.add_argument("--phase2predict", type=str, default='encoding', choices=['encoding', 'delay'],
                        help="Phase to predict")
    parser.add_argument("--standardize_betas", action='store_true', 
                        help="If set, standardizes betas before decoding.")
    parser.add_argument("--standardize_acts", action='store_true', 
                        help="If set, standardizes activations before decoding.")   
    
    # Cache / Preprocessing Arguments
    parser.add_argument("--save_data", action='store_true', 
                        help="If set, saves the concatenated s_betas and s_acts to disk after processing.")
    parser.add_argument("--load_data", action='store_true',
                        help="If set, loads s_betas and s_acts from disk instead of reprocessing raw files.")
    parser.add_argument("--data_cache_dir", type=str, default=None,
                        help="Directory to save/load cached data. Defaults to save_path/run_name if not specified.")

    # ROI list
    default_rois_list = [
        'SFL', 'i6-8', 's6-8', 'IFJa', 'IFJp', 'IFSp', 'IFSa', '8BM', '8Av', '8Ad', '8BL', '8C', 
        '9m', '9p', '9a', '9-46d', 'a9-46v', 'p9-46v', '46', '44', '45', '47l', '47m', '47s', 
        'a47r', 'p47r', '10r', '10d', '10v', 'a10p', 'p10p', '10pp', '11l', '13l'
    ]
    parser.add_argument("--rois", type=str, default=str(default_rois_list), 
                        help="String representation of a python list of ROIs.")
    
    args = parser.parse_args()

    # Define results folder
    folder_path = os.path.join(args.save_path, args.run_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Define Cache Directory
    # If explicit path provided, use it. Otherwise use the results folder.
    cache_dir = args.data_cache_dir if args.data_cache_dir else folder_path

    # --- BLOCK 1: Load or Build Data ---
    betas, acts = build_data(args.behav_dir, args.betas_dir, args.acts_dir, args.subj, args.sessions)
    s_betas, s_acts = select_data(betas, acts, phase2predict=args.phase2predict)
    
    # Save if requested
    if args.save_data:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Saving processed data to: {cache_dir}")
        np.save(os.path.join(cache_dir, 's_betas.npy'), s_betas)
        np.save(os.path.join(cache_dir, 's_acts.npy'), s_acts)

    # --- BLOCK 2: Run ROI Inference ---
    
    # Parse ROI string
    try:
        rois = ast.literal_eval(args.rois)
    except Exception as e:
        print(f"Error parsing ROI string: {e}")
        return

    # Load glasser dlabel info
    dl = nib.load(args.dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    label_dict = dl.header.get_axis(0).label[0] 

    results = {}
    regressors = {}
    print(f"Starting decoding for {len(rois)} ROIs...")
    
    for roi in rois:
        mask = create_beta_mask((label_dict, data), roi, args.lateralize)
        if mask.sum() == 0:
            continue    
        
        curr_betas = s_betas[:, mask]
        
        try:
            result, _ , regressor = predict(
                curr_betas, s_acts, 
                model=pca_ridge_decode, 
                avg_vertices=True, 
                standardize_acts=args.standardize_acts,
                standardize_betas=args.standardize_betas,
                ridge_alpha=0.5, 
                n_pcs=64
            )
            results[roi] = result
            regressors[roi] = regressor
            print(f"Finished {roi}")
        except Exception as e:
            print(f"Failed decoding {roi}: {e}")

    # Save Results
    save_file = os.path.join(folder_path, 'results.json')
    with open(save_file, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {save_file}")

 

if __name__ == "__main__":
    main()