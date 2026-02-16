import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import ast
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------

def get_freq_poly(data, bins=50):
    """
    Compute histogram density and bin centers for frequency polygon plotting.
    Uses density=True to allow comparison of datasets with different N.
    """
    if len(data) == 0:
        return np.array([]), np.array([])
        
    # Density=True normalizes the area under the curve to 1
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts

def main():
    parser = argparse.ArgumentParser(description="Plot True (Global) vs Stacked Preds (Multiple Tasks) Histograms")
    parser.add_argument("--lateralization", type=str, default="LH_RH")
    
    # CHANGED: Now accepts a list string
    parser.add_argument("--ctrl_tasks", type=str, required=True, 
                        help="List of tasks to stack. Format: \"['task-ctxdm_acq-col', 'task-1back_acq-obj']\"")
    
    parser.add_argument("--rois", type=str, default="['10r', '46']")
    parser.add_argument("--base_dir", type=str, default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed")
    
    # Path arguments aligned with scatter script
    parser.add_argument("--dlabel_path", type=str, default="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii")
    parser.add_argument("--decode_results_dir", type=str, 
                        default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay",
                        help="Path containing the 'regressors' folder with saved scalers")
    parser.add_argument("--run_name", type=str, default="128c", help="Name for this inference run (used to locate scalers)")
    
    args = parser.parse_args()
    
    # Parse inputs
    rois = ast.literal_eval(args.rois)
    ctrl_tasks_list = ast.literal_eval(args.ctrl_tasks)
    results_dir = os.path.join(args.base_dir, "results")
    
    # ---------------------------------------------------------
    # 2. Load Data 
    # ---------------------------------------------------------
    
    # A. Load True Betas (Global Reference from NPZ)
    true_betas_path = f"{args.decode_results_dir}/betas.npz"
    
    if not os.path.exists(true_betas_path):
        raise FileNotFoundError(f"True betas file not found: {true_betas_path}")
    
    print(f"Loading Global True Betas from: {true_betas_path}")
    true_betas_dict = np.load(true_betas_path)

    # B. Load and Stack Predicted Betas (Iterate over tasks)
    print(f"\nLoading Predictions for {len(ctrl_tasks_list)} tasks...")
    
    # Container structure: {'ROI_Name': [array_task1, array_task2, ...]}
    stacked_preds_container = {roi: [] for roi in rois}
    
    for task_name in ctrl_tasks_list:
        pred_path = os.path.join(results_dir, f"ctrl_pred_betas/{args.lateralization}/{args.run_name}/{task_name}/results.npz")
        
        if not os.path.exists(pred_path):
            print(f"  [Warning] Missing predictions for: {task_name}")
            continue
            
        try:
            task_data = np.load(pred_path)
            # Append data for each requested ROI
            for region in rois:
                if region in task_data:
                    stacked_preds_container[region].append(task_data[region].reshape(-1))
                else:
                    pass # ROI missing in this specific task file
        except Exception as e:
            print(f"  [Error] Failed to load {task_name}: {e}")

    # 3. Prepare Output Dir
    # We create a specific folder for this comparison to avoid overwriting single-task plots
    save_dir = os.path.join(results_dir, f"ctrl_pred_betas/{args.lateralization}/{args.run_name}/stacked_comparison_plots")
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 3. Processing Loop
    # ---------------------------------------------------------
    stats_results = []

    for region in rois:
        print(f"\nProcessing ROI: {region}")
        
        # --- PROCESS TRUE BETAS ---
        if region not in true_betas_dict:
            print(f"  Warning: ROI {region} not found in True Betas dict. Skipping.")
            continue
            
        roi_tb = true_betas_dict[region].flatten()

        # --- PROCESS PRED BETAS (Stacking) ---
        if not stacked_preds_container[region]:
             print(f"  Warning: No prediction data found for ROI {region} across provided tasks.")
             continue
        
        # Concatenate all lists into one big array
        roi_pb = np.concatenate(stacked_preds_container[region])

        # --- STATS ---
        # Calculate means for the plot title/lines
        mean_true = np.mean(roi_tb)
        mean_pred = np.mean(roi_pb)
        
        print(f"  True(n={len(roi_tb)}) mean={mean_true:.2f} | Pred(n={len(roi_pb)}) mean={mean_pred:.2f}")

        stats_results.append({
            'ROI': region,
            'N_True': len(roi_tb),
            'Mean_True': mean_true,
            'N_Pred_Stacked': len(roi_pb),
            'Mean_Pred': mean_pred,
            'Tasks_Included': len(ctrl_tasks_list)
        })

        # --- PLOTTING: Side-by-Side Histograms (Density Normalized) ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=False) 
        
        fig.suptitle(f"ROI: {region} - Global Truth vs Stacked Predictions ({len(ctrl_tasks_list)} tasks)", fontsize=14, fontweight='bold')

        # Subplot 1: True Betas (Global)
        tb_centers, tb_counts = get_freq_poly(roi_tb)
        ax1.plot(tb_centers, tb_counts, color='green', linewidth=2)
        ax1.fill_between(tb_centers, tb_counts, color='green', alpha=0.1)
        ax1.axvline(mean_true, color='green', linestyle='--', alpha=0.6, label=f'Mean: {mean_true:.2f}')
        ax1.set_title(f"True Betas (Global Training Set)\nn={len(roi_tb)}")
        ax1.set_ylabel("Density")
        ax1.set_xlabel("Beta Value")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Subplot 2: Predicted Betas (Stacked)
        pb_centers, pb_counts = get_freq_poly(roi_pb)
        ax2.plot(pb_centers, pb_counts, color='purple', linewidth=2)
        ax2.fill_between(pb_centers, pb_counts, color='purple', alpha=0.1)
        ax2.axvline(mean_pred, color='purple', linestyle='--', alpha=0.6, label=f'Mean: {mean_pred:.2f}')
        ax2.set_title(f"Predicted Betas (Stacked)\nn={len(roi_pb)}")
        ax2.set_ylabel("Density")
        ax2.set_xlabel("Beta Value")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        
        plot_name = f"dist_stacked_{region}.png"
        plt.savefig(os.path.join(save_dir, plot_name))
        plt.close()

    # ---------------------------------------------------------
    # 4. Save Stats
    # ---------------------------------------------------------
    if stats_results:
        df_stats = pd.DataFrame(stats_results)
        csv_path = os.path.join(save_dir, "stacked_stats.csv")
        df_stats.to_csv(csv_path, index=False)
        print(f"\nStats saved to: {csv_path}")

    print(f"All plots saved to: {save_dir}")

if __name__ == "__main__":
    main()