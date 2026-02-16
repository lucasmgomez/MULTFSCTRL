import os
import numpy as np
import torch
import json
import argparse
import ast
from joblib import load
from models.regression import pca_ridge_infer

# --- HELPER from predict_arg.py (Must match exactly) ---
def tc_format(task, tc):
    task_tc_len_map = {'ctxdm': 6, 'interdms': 8, '1back': 12} 
    correct_len = task_tc_len_map.get(task, len(str(tc))) 
    if len(str(tc)) < correct_len:
        tc = tc.zfill(correct_len)   
    if task == '1back': 
        tc = tc[:10]
    return tc         

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

def get_activations_sorted(acts_fp, task_name):
    """
    Loads activations and sorts them by TC to match predict_arg.py order.
    """
    acts_dict = torch.load(acts_fp, map_location='cpu') 
    
    # 1. Load Raw
    raw_acts = acts_dict['layer_activations'].to(torch.float32) # Shape: (Layers, TCs, Tokens, Units) ?
    # Note: Check your shape. Usually it's (Layers, TCs, ...) based on predict_arg usage.
    
    raw_tcs = acts_dict['tcs'] # List of TCs corresponding to dim 1
    
    # 2. Format TCs (handle integers, padding, etc.)
    fmt_tcs = [tc_format(task_name, str(tc)) for tc in raw_tcs]
    
    # 3. Get Sort Indices (A-Z)
    # This matches `sorted(betas.keys())` in predict_arg.py
    sort_idx = np.argsort(fmt_tcs)
    
    # 4. Reorder Activations
    # Assuming dim 1 is the batch/TC dimension based on: acts[:, all_tcs.index(tc)]
    sorted_acts = raw_acts[:, sort_idx] 
    
    print(f"Sorted {len(fmt_tcs)} activations. First 3 TCs: {[fmt_tcs[i] for i in sort_idx[:3]]}")
    
    return sorted_acts.numpy(), sort_idx

def select_acts(acts, n_images=5, phase='delay'):
    # ... (Same as before) ...
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

    # ... (Same args as before) ...
    parser.add_argument("--decode_results_dir", type=str, required=True)
    parser.add_argument("--acts_pth", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="results/test_ctrl")
    parser.add_argument("--run_name", type=str, default="test_infer")
    parser.add_argument("--task_length", type=int, default=5)
    parser.add_argument("--phase", type=str, default="delay")
    parser.add_argument("--standardize_acts", action='store_true')   
    parser.add_argument("--rois", type=str, default="['10r', '46']")
    parser.add_argument("--pca", action='store_true', help="If set, applies PCA to activations before decoding.")
    
    args = parser.parse_args()

    # EXTRACT TASK NAME for TC formatting
    # Assuming run_name or filename contains the task (e.g., "task-1back")
    if "1back" in args.run_name: task = "1back"
    elif "ctxdm" in args.run_name: task = "ctxdm"
    elif "interdms" in args.run_name: task = "interdms"
    else: task = "unknown" # Fallback

    # LOAD AND SORT ACTIVATIONS
    print(f"Loading activations from {args.acts_pth} (Task: {task})...")
    acts, _ = get_activations_sorted(args.acts_pth, task)
    
    # Select Phase/Tokens
    s_acts, time_dim = select_acts(acts, n_images=args.task_length, phase=args.phase)
    
    # Parse ROIs
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
        blayer = get_best_layer(decode_results, roi)
        roi_blayer_acts = s_acts[blayer]

        # Get regressor
        reg_dir = os.path.join(args.decode_results_dir, f'regressors/{roi}')
        regressor = load(os.path.join(reg_dir, f'layer_{blayer}.joblib'))
        scalar = load(os.path.join(reg_dir, f'layer_{blayer}_scalar.joblib'))
        
        if args.pca:
            pca_fp = os.path.join(args.decode_results_dir, f'regressors/{roi}/layer_{blayer}_pca.joblib')
            pca = load(pca_fp)
        else: 
            pca = None

        try:
            pred_betas = pca_ridge_infer(
                roi_blayer_acts, 
                regressor=regressor, 
                pca=pca,
                scalar=scalar,
                standardize_acts=args.standardize_acts,
            )
        except Exception as e:
            print(f"Failed decoding {roi}: {e}")
            continue

        # Reshape into time dim
        # Final Shape: (N_TCs, Time, Vertices)
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