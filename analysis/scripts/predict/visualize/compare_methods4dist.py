import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import ast

def get_freq_poly(data, bins=30):
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts

def parse_beta_string(beta_str):
    """
    Parses a string representation of a list/array from CSV back into a list of floats.
    Handles both comma-separated (standard list) and space-separated (numpy str) formats.
    """
    try:
        # Try standard list format first: [-0.1, 0.2, ...]
        return ast.literal_eval(beta_str)
    except (ValueError, SyntaxError):
        # Fallback for space-separated format: [ -0.1  0.2 ]
        clean_str = beta_str.replace('[', '').replace(']', '').replace('\n', ' ')
        return [float(x) for x in clean_str.split() if x.strip()]

def main():
    parser = argparse.ArgumentParser(description="Plot beta distributions per ROI")
    parser.add_argument("--lateralization", type=str, default="LH")
    parser.add_argument("--rois", type=str, default="['10r', '46']")
    parser.add_argument("--base_dir", type=str, default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results")
    parser.add_argument("--betas_type", type=str, default="frame-only_enc+delay_delay_lsa_wfdelay")
    
    # NEW ARGUMENT for the CSV file
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the ROI betas CSV file")
    
    args = parser.parse_args()
    rois = ast.literal_eval(args.rois)
    
    # ---------------------------------------------------------
    # 1. Load and Filter "True Betas" from CSV
    # ---------------------------------------------------------
    print(f"Loading true betas from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    # ---------------------------------------------------------
    # 2. Load "Predicted Betas" (Kept as is)
    # ---------------------------------------------------------
    base_betas_path = os.path.join(args.base_dir, f"{args.betas_type}/betas.npz")
    print(f"Loading betas from: {base_betas_path}")
    base_betas = np.load(base_betas_path)
    
    save_dir = os.path.join(args.base_dir, f"{args.betas_type}/plots_compare_method")
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 3. Plotting Loop
    # ---------------------------------------------------------
    for region in rois:
        print(f"Processing ROI: {region}")
        
        # --- A. Extract True Betas for this ROI ---
        roi_df = df[df['roi'] == region]
        
        if roi_df.empty:
            print(f"  Warning: No data found for ROI {region} in CSV. Skipping.")
            continue

        # Collect all lists from the 'betas' column and flatten them into one big array
        # We assume the CSV 'betas' column contains strings like "[-0.12, 0.97, ...]"
        extracted_values = []
        for val_str in roi_df['betas']:
            parsed_list = parse_beta_string(str(val_str))
            extracted_values.extend(parsed_list)
            
        roi_cb = np.array(extracted_values)

        # --- B. Extract Pred Betas ---
        if region not in base_betas:
             print(f"  Warning: ROI {region} not found in predicted .npz file. Skipping.")
             continue
             
        roi_bb = base_betas[region]

        print(region, "CSV len", roi_cb.shape, "NPZ len", roi_bb.shape)
        print(region, "CSV mean/std", roi_cb.mean(), roi_cb.std())
        print(region, "NPZ mean/std", roi_bb.mean(), roi_bb.std())
        print(region, "corr(CSV, NPZ) if same length:", 
              np.corrcoef(roi_cb[:len(roi_bb)], roi_bb[:len(roi_cb)])[0,1] if len(roi_cb)==len(roi_bb) else "n/a")

        # --- C. Plotting ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
        fig.suptitle(f"ROI: {region}\nDistribution Comparison", fontsize=14)

        # Subplot 1: CSV Betas
        cb_centers, cb_counts = get_freq_poly(roi_cb)
        ax1.plot(cb_centers, cb_counts, color='green', linewidth=2)
        ax1.axvline(roi_cb.mean(), color='green', linestyle='--', alpha=0.6, label=f'Mean: {roi_cb.mean():.4f}')
        ax1.set_title(f"CSV Betas (n={len(roi_cb)})")
        ax1.set_ylabel("Frequency (Counts)")
        ax1.set_xlabel("Value")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Subplot 2: BASE Betas (from NPZ)
        pb_centers, pb_counts = get_freq_poly(roi_bb)
        ax2.plot(pb_centers, pb_counts, color='purple', linewidth=2)
        ax2.axvline(roi_bb.mean(), color='purple', linestyle='--', alpha=0.6, label=f'Mean: {roi_bb.mean():.4f}')
        ax2.set_title("Base Betas")
        ax2.set_ylabel("Frequency (Counts)")
        ax2.set_xlabel("Value")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_name = f"dist_{region}.png"
        plt.savefig(os.path.join(save_dir, plot_name))
        plt.close()

    print(f"All plots saved to: {save_dir}")

if __name__ == "__main__":
    main()