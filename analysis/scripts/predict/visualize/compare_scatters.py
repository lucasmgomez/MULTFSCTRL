import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import ast
import nibabel as nib
import pandas as pd
import glob
from joblib import load
from scipy.stats import pearsonr

# ---------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------
def load_atlas_data(dlabel_path):
    dl = nib.load(dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    label_dict = dl.header.get_axis(0).label[0] 
    return label_dict, data

def create_roi_mask(dlabel_info, roi_name, lateralize='LH_RH'):
    label_dict, data = dlabel_info
    target_labels = []
    
    if lateralize == 'LH_RH':
        target_labels = [f'L_{roi_name}_ROI', f'R_{roi_name}_ROI']
    elif lateralize == 'LH':
        target_labels = [f'L_{roi_name}_ROI']
    elif lateralize == 'RH':
        target_labels = [f'R_{roi_name}_ROI']
    else:
        target_labels = [roi_name]

    matched_keys = [k for k, (name, _) in label_dict.items() if name in target_labels]
    mask = np.isin(data, matched_keys)
    return mask

def main():
    parser = argparse.ArgumentParser(description="Plot True vs Pred Scatter per ROI")
    parser.add_argument("--lateralization", type=str, default="LH_RH")
    parser.add_argument("--ctrl_task", type=str, required=True, help="Input format: task-name_acqlabel")
    parser.add_argument("--rois", type=str, default="['10r', '46']")
    parser.add_argument("--base_dir", type=str, default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed")
    
    parser.add_argument("--dlabel_path", type=str, default="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii")
    parser.add_argument("--decode_results_dir", type=str, 
                        default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay",
                        help="Path containing the 'regressors' folder with saved scalers")
    parser.add_argument("--run_name", type=str, default="128c", help="Name for this inference run (used to locate scalers)")
    
    args = parser.parse_args()
    rois = ast.literal_eval(args.rois)
    results_dir = os.path.join(args.base_dir, "results")
    
    # ---------------------------------------------------------
    # 1. ROBUST TASK MAPPING
    # ---------------------------------------------------------
    if "acq-" in args.ctrl_task:
        mapped_ctrl_task_str = args.ctrl_task
    else:
        parts = args.ctrl_task.split('_', 1) 
        if len(parts) > 1:
            task_base = parts[0]
            acq_suffix = parts[1]
            mapped_ctrl_task_str = f"{task_base}_acq-{acq_suffix}"
        else:
            mapped_ctrl_task_str = args.ctrl_task 

    print(f"Mapped Task Name: {mapped_ctrl_task_str}")

    # ---------------------------------------------------------
    # 2. Load Data (Concatenating All Runs)
    # ---------------------------------------------------------
    
    # A. Load True Betas (Full Brain)
    search_pattern = os.path.join(args.base_dir, f"preprocessed_data/{mapped_ctrl_task_str}*_selected_betas.npy")
    found_files = sorted(glob.glob(search_pattern))
    
    if not found_files:
        raise FileNotFoundError(f"No true beta files found matching: {search_pattern}")
    
    print(f"Found {len(found_files)} true beta files. Loading...")
    loaded_arrays = [np.load(f) for f in found_files]
    full_brain_betas = np.concatenate(loaded_arrays, axis=0)

    # B. Load Predicted Betas (ROI specific)
    pred_betas_path = os.path.join(results_dir, f"ctrl_pred_betas/{args.lateralization}/{args.run_name}/{args.ctrl_task}/results.npz")
    print(f"Loading pred betas from: {pred_betas_path}")
    if not os.path.exists(pred_betas_path):
        raise FileNotFoundError(f"Prediction file missing: {pred_betas_path}")
        
    pred_betas = np.load(pred_betas_path)

    # 3. Prepare Output Dir
    save_dir = os.path.join(results_dir, f"ctrl_pred_betas/{args.lateralization}/{args.run_name}/{args.ctrl_task}/plots_scatter")
    os.makedirs(save_dir, exist_ok=True)

    # 4. Load Atlas
    atlas_info = load_atlas_data(args.dlabel_path)

    # ---------------------------------------------------------
    # 3. Processing Loop
    # ---------------------------------------------------------
    correlation_results = []

    for region in rois:
        print(f"\nProcessing ROI: {region}")
        
        # --- PROCESS TRUE BETAS ---
        mask = create_roi_mask(atlas_info, region, args.lateralization) 
        if mask.sum() == 0:
            print(f"Warning: ROI {region} not found in atlas. Skipping.")
            continue
            
        roi_raw_values = full_brain_betas[:, mask] 
        roi_avg_values = np.mean(roi_raw_values, axis=1) 

        # Apply Scaler
        scaler_path = os.path.join(args.decode_results_dir, "regressors", region, "betas_scalar.joblib")
        if not os.path.exists(scaler_path):
            print(f"Warning: Scaler not found for {region} at {scaler_path}. Skipping.")
            continue
            
        scaler = load(scaler_path)
        roi_tb = scaler.transform(roi_avg_values.reshape(-1, 1)).flatten()

        # --- PROCESS PRED BETAS ---
        if region not in pred_betas:
             print(f"Warning: ROI {region} not found in predictions. Skipping.")
             continue
        roi_pb = pred_betas[region]
        roi_pb_reshaped = roi_pb.reshape(-1) 

        # --- STATS ---
        corr_title_str = "r=N/A"
        if len(roi_tb) == len(roi_pb_reshaped):
            r_val, p_val = pearsonr(roi_tb, roi_pb_reshaped)
            print(f">> ROI {region} Correlation: r = {r_val:.4f} (p = {p_val:.4e})")
            
            p_str = "<0.001" if p_val < 0.001 else f"={p_val:.3f}"
            corr_title_str = f"r={r_val:.3f}, p{p_str}"

            correlation_results.append({
                'ROI': region,
                'Task': args.ctrl_task,
                'Pearson_R': r_val,
                'P_Value': p_val,
                'N_Samples': len(roi_tb)
            })
        else:
            print(f">> ROI {region} Correlation Skipped: Shape mismatch")

        # --- PLOTTING: True vs Pred Scatter ---
        plt.figure(figsize=(8, 8))
        
        # Scatter Plot
        # Using a single color since X vs Y defines the relationship
        plt.scatter(roi_tb, roi_pb_reshaped, color='purple', alpha=0.5, s=30, edgecolors='white', linewidth=0.5)
        
        # Add Reference Line (y=x)
        min_val = min(roi_tb.min(), roi_pb_reshaped.min())
        max_val = max(roi_tb.max(), roi_pb_reshaped.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label="Perfect Fit")
        
        plt.title(f"Task: {args.ctrl_task} - ROI: {region}\n{corr_title_str}", fontsize=14, fontweight='bold')
        plt.xlabel("True Betas (Z-Scored)")
        plt.ylabel("Predicted Betas")
        plt.legend()
        plt.grid(alpha=0.3)

        # Plot Saving
        plot_name = f"scatter_corr_{region}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, plot_name))
        plt.close()

    # ---------------------------------------------------------
    # 4. Save Stats
    # ---------------------------------------------------------
    if correlation_results:
        df_corr = pd.DataFrame(correlation_results)
        csv_path = os.path.join(save_dir, "correlation_results.csv")
        df_corr.to_csv(csv_path, index=False)
        print(f"\nCorrelation results saved to: {csv_path}")

    print(f"All plots saved to: {save_dir}")

if __name__ == "__main__":
    main()