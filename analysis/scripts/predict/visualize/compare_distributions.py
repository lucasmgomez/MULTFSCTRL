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

def get_freq_poly(data, bins=30):
    """
    Compute histogram counts and bin centers for frequency polygon plotting.
    """
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts

def main():
    parser = argparse.ArgumentParser(description="Plot True (All Tasks) vs Pred (Ctrl Task) Histograms per ROI")
    parser.add_argument("--lateralization", type=str, default="LH_RH")
    parser.add_argument("--ctrl_task", type=str, required=True, help="Input format: task-name_acqlabel")
    parser.add_argument("--rois", type=str, default="['10r', '46']")
    parser.add_argument("--base_dir", type=str, default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed")
    
    # Path arguments aligned with scatter script
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
    # 2. Load Data 
    # ---------------------------------------------------------
    
    # A. Load True Betas (ALL TASKS from specific .npz)
    # Hardcoded path as requested
    true_betas_path = f"{args.decode_results_dir}/betas.npz"
    
    if not os.path.exists(true_betas_path):
        raise FileNotFoundError(f"True betas file not found: {true_betas_path}")
    
    print(f"Loading ALL TASKS true betas from: {true_betas_path}")
    true_betas_dict = np.load(true_betas_path)

    # B. Load Predicted Betas (ROI specific, single task)
    pred_betas_path = os.path.join(results_dir, f"ctrl_pred_betas/{args.lateralization}/{args.run_name}/{args.ctrl_task}/results.npz")
    print(f"Loading pred betas from: {pred_betas_path}")
    if not os.path.exists(pred_betas_path):
        raise FileNotFoundError(f"Prediction file missing: {pred_betas_path}")
        
    pred_betas = np.load(pred_betas_path)

    # 3. Prepare Output Dir
    save_dir = os.path.join(results_dir, f"ctrl_pred_betas/{args.lateralization}/{args.run_name}/{args.ctrl_task}/plots_hist")
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 3. Processing Loop
    # ---------------------------------------------------------
    correlation_results = []

    for region in rois:
        print(f"\nProcessing ROI: {region}")
        
        # --- PROCESS TRUE BETAS ---
            
        roi_tb = true_betas_dict[region]
        roi_tb = roi_tb.flatten()

        # --- PROCESS PRED BETAS ---
        if region not in pred_betas:
             print(f"Warning: ROI {region} not found in predictions. Skipping.")
             continue
        roi_pb = pred_betas[region]
        roi_pb_reshaped = roi_pb.reshape(-1) 

        # --- STATS ---
        # Only calculate R if sample sizes match (Unlikely here, but kept for logic safety)
        corr_title_str = "Shapes differ (Dist Only)"
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
            print(f">> ROI {region} Stats Skipped: Shape mismatch (True(All): {len(roi_tb)}, Pred(Task): {len(roi_pb_reshaped)})")

        # --- PLOTTING: Histogram Overlay ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=False) # Changed sharex to False as ranges might differ significantly
        
        fig.suptitle(f"Task: {args.ctrl_task} - ROI: {region}\n{corr_title_str}", fontsize=14, fontweight='bold')

        # Subplot 1: True Betas (All Tasks)
        tb_centers, tb_counts = get_freq_poly(roi_tb)
        ax1.plot(tb_centers, tb_counts, color='green', linewidth=2)
        ax1.fill_between(tb_centers, tb_counts, color='green', alpha=0.1)
        ax1.axvline(roi_tb.mean(), color='green', linestyle='--', alpha=0.6, label=f'Mean: {roi_tb.mean():.4f}')
        ax1.set_title(f"True Betas (All Tasks, n={len(roi_tb)})")
        ax1.set_ylabel("Frequency")
        ax1.set_xlabel("Value")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Subplot 2: Predicted Betas (Specific Task)
        pb_centers, pb_counts = get_freq_poly(roi_pb_reshaped)
        ax2.plot(pb_centers, pb_counts, color='purple', linewidth=2)
        ax2.fill_between(pb_centers, pb_counts, color='purple', alpha=0.1)
        ax2.axvline(roi_pb_reshaped.mean(), color='purple', linestyle='--', alpha=0.6, label=f'Mean: {roi_pb_reshaped.mean():.4f}')
        ax2.set_title(f"Predicted Betas (Pred Task Only, n={len(roi_pb_reshaped)})")
        ax2.set_ylabel("Frequency")
        ax2.set_xlabel("Value")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        
        plot_name = f"dist_{region}.png"
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