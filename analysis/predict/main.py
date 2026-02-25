import json
import argparse
import ast
import os
import numpy as np
import nibabel as nib
from joblib import dump
from models.regression import pls_decode, pca_ridge_decode

from predict import build_data, select_data, create_beta_mask, predict

def main():
    parser = argparse.ArgumentParser(description="fMRI Decoding")

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
    
    # Model configuration
    parser.add_argument("--model_type", type=str, choices=['pls', 'pca_ridge'], default='pca_ridge',
                        help="Which decoding model to use")
    
    # Run configuration
    parser.add_argument("--run_name", type=str, default="test_run", help="Name of the current run/experiment")
    parser.add_argument("--subj", type=str, default="sub-01", help="Subject ID")
    parser.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02", "ses-03", "ses-04"],
                        help="List of sessions (space separated)")
    parser.add_argument("--events_type", type=str, default='wfdelay', choices=['wfdelay', 'wofdelay'], 
                        help="Type of events to use for building data")
    parser.add_argument("--lateralize", type=str, choices=['LR', 'L', 'R'], default='LR',
                        help="Lateralization of ROIs")
    parser.add_argument("--phase2predict", type=str, default='delay', choices=['encoding', 'delay'],
                        help="Phase to predict")
    parser.add_argument("--standardize_betas", action='store_true', 
                        help="If set, standardizes betas before decoding.")
    parser.add_argument("--standardize_acts", action='store_true', 
                        help="If set, standardizes activations before decoding.")   
    
    # Model Hyperparameters
    parser.add_argument("--pls_n_pcs", type=str, default="[10, 20, 30, 40, 50]",
                        help="String representation of a list of n_components to try for PLS.")
    parser.add_argument("--ridge_alphas", type=str, default="[0.1, 1.0, 10.0]", 
                        help="String representation of a list of alpha values for RidgeCV.")
    parser.add_argument("--ridge_n_pcs", type=int, default=128, 
                        help="Number of PCA components for Ridge regression")
    
    # Cache / Preprocessing Arguments
    parser.add_argument("--save_data", action='store_true', 
                        help="If set, saves the concatenated s_betas and s_acts to disk after processing.")
    parser.add_argument("--load_data", action='store_true',
                        help="If set, loads s_betas and s_acts from disk instead of reprocessing raw files.")
    parser.add_argument("--data_cache_dir", type=str, default=None,
                        help="Directory to save/load cached data.")
    parser.add_argument("--save_per_run", action='store_true',
                        help="If set, saves individual .npy files for each run's selected betas/acts.")

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
    
    cache_dir = args.data_cache_dir if args.data_cache_dir else folder_path

    # --- BLOCK 1: Load or Build Data ---
    if args.load_data:
        print(f"Loading preprocessed data from: {cache_dir}")
        s_betas = np.load(os.path.join(cache_dir, 's_betas.npy'))
        s_acts = np.load(os.path.join(cache_dir, 's_acts.npy'))
        print(f"Data loaded successfully. Shapes: Betas {s_betas.shape}, Acts {s_acts.shape}")

    else:
        print(f"Processing Subject: {args.subj}")
        print(f"Sessions: {args.sessions}")
        
        betas, acts = build_data(args.behav_dir, args.betas_dir, args.acts_dir, args.subj, args.sessions, events_type=args.events_type)
        s_betas, s_acts = select_data(
            betas, acts, 
            phase2predict=args.phase2predict, 
            save_per_run=args.save_per_run, 
            cache_dir=cache_dir
        )
        
        if args.save_data:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Saving processed data to: {cache_dir}")
            np.save(os.path.join(cache_dir, 's_betas.npy'), s_betas)
            np.save(os.path.join(cache_dir, 's_acts.npy'), s_acts)

    # --- BLOCK 2: Run ROI Decoding ---
    rois = ast.literal_eval(args.rois)

    dl = nib.load(args.dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    label_dict = dl.header.get_axis(0).label[0] 

    results = {}
    modules = {}
    bscalars = {}
    betas_to_save = {}
    print(f"Starting decoding for {len(rois)} ROIs using {args.model_type.upper()}...")
    
    for roi in rois:
        mask = create_beta_mask((label_dict, data), roi, args.lateralize)
        if mask.sum() == 0:
            continue    
        
        curr_betas = s_betas[:, mask]
        
        if args.model_type == 'pls':
            pls_n_pcs = [float(a) for a in ast.literal_eval(args.pls_n_pcs)]
            res = predict(
                curr_betas, s_acts, 
                model=pls_decode, 
                avg_vertices=True, 
                standardize_acts=args.standardize_acts,
                standardize_betas=args.standardize_betas,
                n_pcs=pls_n_pcs
            )
            result, _, regressor, scalar, curr_betas_, curr_betas_scalar = res
            modules[roi] = (regressor, scalar)
            
        elif args.model_type == 'pca_ridge':
            ridge_alphas = [float(a) for a in ast.literal_eval(args.ridge_alphas)]
            res = predict(
                curr_betas, s_acts, 
                model=pca_ridge_decode, 
                avg_vertices=True, 
                standardize_acts=args.standardize_acts,
                standardize_betas=args.standardize_betas,
                ridge_alphas=ridge_alphas, 
                n_pcs=args.ridge_n_pcs
            )
            result, _, regressor, pca, scalar, curr_betas_, curr_betas_scalar = res
            modules[roi] = (regressor, pca, scalar)

        results[roi] = result
        bscalars[roi] = curr_betas_scalar
        betas_to_save[roi] = curr_betas_
        print(f"Finished {roi}")

    # Save Results
    save_file = os.path.join(folder_path, 'results.json')
    with open(save_file, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {save_file}")

    # Save Regressors
    regressors_path = os.path.join(folder_path, 'regressors')
    for k, mods in modules.items():
        regressor_path = os.path.join(regressors_path, k)
        os.makedirs(regressor_path, exist_ok=True)
 
        if args.model_type == 'pls':
            for layer_idx, (reg, scalar) in enumerate(zip(*mods)):
                dump(reg, os.path.join(regressor_path, f'layer_{layer_idx}.joblib'))
                dump(scalar, os.path.join(regressor_path, f'layer_{layer_idx}_scalar.joblib'))
        elif args.model_type == 'pca_ridge':
            for layer_idx, (reg, pca, scalar) in enumerate(zip(*mods)):
                dump(reg, os.path.join(regressor_path, f'layer_{layer_idx}.joblib'))
                dump(pca, os.path.join(regressor_path, f'layer_{layer_idx}_pca.joblib'))
                dump(scalar, os.path.join(regressor_path, f'layer_{layer_idx}_scalar.joblib'))

        bscalar = bscalars[k]
        dump(bscalar, os.path.join(regressor_path, f'betas_scalar.joblib'))

    # Save Betas as npz
    betas_file = os.path.join(folder_path, 'betas.npz')
    np.savez(betas_file, **betas_to_save)
    print(f"Betas saved to {betas_file}")
    
if __name__ == "__main__":
    main()