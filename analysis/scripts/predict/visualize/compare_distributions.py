import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import ast

def get_freq_poly(data, bins=30):
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts

def main():
    # ... [Keep your existing argument parser and loading logic here] ...
    parser = argparse.ArgumentParser(description="Plot beta distributions per ROI")
    parser.add_argument("--lateralization", type=str, default="LH")
    parser.add_argument("--ctrl_task", type=str, required=True)
    parser.add_argument("--rois", type=str, default="['10r', '46']")
    parser.add_argument("--base_dir", type=str, default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results")
    parser.add_argument("--betas_type", type=str, default="frame-only_enc+delay_delay_lsa_wfdelay")
    
    args = parser.parse_args()
    rois = ast.literal_eval(args.rois)
    
    true_betas = np.load(os.path.join(args.base_dir, f"{args.betas_type}/betas.npz"))
    pred_betas_path = os.path.join(args.base_dir, f"ctrl_pred_betas/{args.lateralization}/{args.ctrl_task}/results.npz")
    pred_betas = np.load(pred_betas_path)
    save_dir = os.path.join(args.base_dir, f"ctrl_pred_betas/{args.lateralization}/{args.ctrl_task}/plots")
    os.makedirs(save_dir, exist_ok=True)

    for region in rois:
        print(f"Generating plot for ROI: {region}")
        
        roi_tb = true_betas[region]
        roi_pb = pred_betas[region]
        roi_pb_reshaped = roi_pb.reshape(-1, roi_pb.shape[-1]) # Simplified reshape

        # Create a figure with two subplots side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
        fig.suptitle(f"Task: {args.ctrl_task} - ROI: {region}\nDistribution Comparison", fontsize=14)

        # Subplot 1: True Betas
        tb_centers, tb_counts = get_freq_poly(roi_tb)
        ax1.plot(tb_centers, tb_counts, color='green', linewidth=2)
        ax1.axvline(roi_tb.mean(), color='green', linestyle='--', alpha=0.6, label=f'Mean: {roi_tb.mean():.4f}')
        ax1.set_title("True Betas")
        ax1.set_ylabel("Frequency (Counts)")
        ax1.set_xlabel("Value")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Subplot 2: Predicted Betas
        pb_centers, pb_counts = get_freq_poly(roi_pb_reshaped)
        ax2.plot(pb_centers, pb_counts, color='purple', linewidth=2)
        ax2.axvline(roi_pb_reshaped.mean(), color='purple', linestyle='--', alpha=0.6, label=f'Mean: {roi_pb_reshaped.mean():.4f}')
        ax2.set_title("Predicted Betas")
        ax2.set_ylabel("Frequency (Counts)")
        ax2.set_xlabel("Value")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        
        plot_name = f"dist_{region}.png"
        plt.savefig(os.path.join(save_dir, plot_name))
        plt.close()

    print(f"All plots saved to: {save_dir}")

if __name__ == "__main__":
    main()