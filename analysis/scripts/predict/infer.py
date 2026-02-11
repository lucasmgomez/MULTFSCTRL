import os
import numpy as np
import torch
import json
import argparse
import ast
from joblib import load
from models.regression import pca_ridge_infer

def get_best_layer(decode_results, roi):
    max_r = -1.0
    best_layer = "-1"
    roi_results = decode_results[roi]
    for layer in roi_results:
        layer_r = roi_results[layer]['r']
        if layer_r > max_r:
            max_r = layer_r
            best_layer = layer
    return int(best_layer)

def get_activations(acts_fp):
    acts_dict = torch.load(acts_fp, map_location='cpu') 
    all_acts = acts_dict['layer_activations'].to(torch.float32)
    trial_idxs = acts_dict['trial_idxs']
    return all_acts.numpy(), trial_idxs

def select_acts(acts, n_images=5, phase='delay'):
    n_tokens = acts.shape[2]
    start_idx = max(0, n_tokens - n_images)
    selected_acts = acts[:, :, start_idx:] 

    encoding_idxs = [i for i in range(0, selected_acts.shape[2], 2)]
    delay_idxs = [i for i in range(1, selected_acts.shape[2], 2)]

    if phase == 'delay':
        encoding_idxs = encoding_idxs[:len(delay_idxs)]

    enc_acts = selected_acts[:, :, encoding_idxs, :]
    delay_acts = selected_acts[:, :, delay_idxs, :]
    selected_acts = np.concatenate((enc_acts, delay_acts), axis=-1)

    time_dim = selected_acts.shape[2]
    selected_acts = selected_acts.reshape(selected_acts.shape[0], selected_acts.shape[1]*selected_acts.shape[2], -1)

    return selected_acts, time_dim

def main():
    parser = argparse.ArgumentParser(description="fMRI PCA Ridge Decoding")

    # Path arguments
    parser.add_argument("--decode_results_dir", type=str, 
                        default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay", 
                        help="Path to decoding results directory")
    parser.add_argument("--acts_pth", type=str, default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/activations/task-ctxDM_LOL.pth",
                        help="Path to activations file")
    parser.add_argument("--save_path", type=str, default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/test_ctrl",
                        help="Root path to save results")
    parser.add_argument("--run_name", type=str, default="test_infer",
                        help="Name for this inference run")
    parser.add_argument("--task_length", type=int, default=5, help="Number of images in task")
    parser.add_argument("--phase", type=str, default="delay", help="Phase to select activations based on: encoding/delay")
    parser.add_argument("--standardize_acts", action='store_true', 
                        help="If set, standardizes activations before decoding.")   

    # ROI list
    default_rois_list = [
        '10r', '46'
    ]
    parser.add_argument("--rois", type=str, default=str(default_rois_list), 
                        help="String representation of a python list of ROIs.")
    
    args = parser.parse_args()

    # Load Acts
    acts, trial_idxs = get_activations(args.acts_pth)
    s_acts, time_dim = select_acts(acts, n_images=args.task_length, phase=args.phase)
    
    # Parse ROI string
    try:
        rois = ast.literal_eval(args.rois)
    except Exception as e:
        print(f"Error parsing ROI string: {e}")
        return

    decode_results_fp = os.path.join(args.decode_results_dir, f'results.json')
    decode_results = json.load(open(decode_results_fp, 'r'))

    print(f"Starting inference for {len(rois)} ROIs...")
    
    results = {}
    for roi in rois:
        # Get best layer
        blayer = get_best_layer(decode_results, roi)
        roi_blayer_acts = s_acts[blayer]

        # Get regressor
        regressor_fp = os.path.join(args.decode_results_dir, f'regressors/{roi}/layer_{blayer}.joblib')
        pca_fp = os.path.join(args.decode_results_dir, f'regressors/{roi}/layer_{blayer}_pca.joblib')
        scalar_fp = os.path.join(args.decode_results_dir, f'regressors/{roi}/layer_{blayer}_scalar.joblib')
        regressor =  load(regressor_fp)
        pca = load(pca_fp)
        scalar = load(scalar_fp)

        try:
            
            pred_betas = pca_ridge_infer(
                roi_blayer_acts, 
                regressor=regressor, 
                pca=pca,
                scalar=scalar,
                standardize_acts=args.standardize_acts,
            )

            print(f"Finished {roi}")
        except Exception as e:
            print(f"Failed decoding {roi}: {e}")

        # Reshape into time dim
        pred_betas = pred_betas.reshape(pred_betas.shape[0]//time_dim, time_dim, -1)
        results[roi] = pred_betas

    # Save Results
    folder_path = os.path.join(args.save_path, args.run_name)
    os.makedirs(folder_path, exist_ok=True)
    save_file = os.path.join(folder_path, 'results.npz')
    np.savez(save_file, **results)
    print(f"Results saved to {save_file}")

 

if __name__ == "__main__":
    main()