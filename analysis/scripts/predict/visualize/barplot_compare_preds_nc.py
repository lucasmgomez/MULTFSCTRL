import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Helper Functions for Noise Ceilings
# ---------------------------------------------------------

def clean_roi_label(roi_str):
    """
    Converts CSV labels like 'L_10pp_ROI+R_10pp_ROI' or 'L_SFL_ROI'
    into clean labels like '10pp' or 'SFL' to match the JSON keys.
    """
    # If it's a bilateral string like "L_X+R_X", just take the first part "L_X"
    part = roi_str.split("+")[0]
    
    # Remove L_ or R_ prefix
    part = re.sub(r"^[LR]_", "", part)
    
    # Remove _ROI suffix
    part = re.sub(r"_ROI$", "", part)
    
    return part

def load_noise_ceilings(csv_path):
    """
    Loads the noise ceiling CSV, cleans ROI names, and returns a dict
    mapping {roi_name: mean_noise_ceiling}.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Noise ceiling CSV not found: {csv_path}")

    # Determine separator (guess TSV if .tsv, else CSV)
    sep = "\t" if csv_path.endswith(".tsv") else ","
    df = pd.read_csv(csv_path, sep=sep)

    # clean ROI names
    df["roi_clean"] = df["roi"].apply(clean_roi_label)

    # Average noise_ceiling across all available rows (tasks, phases, runs) for that ROI
    # If you only want specific phases (e.g. only "Delay"), filter df here before grouping.
    roi_ceilings = df.groupby("roi_clean")["noise_ceiling"].mean().to_dict()

    return roi_ceilings

# ---------------------------------------------------------
# 2. Existing JSON Loading Logic
# ---------------------------------------------------------

def load_max_r_by_roi(json_path, anatomical_order):
    with open(json_path, "r") as f:
        data = json.load(f)

    roi_to_maxr = {}
    for roi in anatomical_order:
        if roi not in data:
            continue

        # Expect data[roi] is a dict of models/settings -> metrics dict containing 'r'
        r_values = []
        for _, metrics in data[roi].items():
            if isinstance(metrics, dict) and "r" in metrics and metrics["r"] is not None:
                r_values.append(metrics["r"])

        if len(r_values) > 0:
            roi_to_maxr[roi] = max(r_values)

    return roi_to_maxr

# ---------------------------------------------------------
# 3. Main Plotting Function
# ---------------------------------------------------------

def plot_roi_comparison(
    json_path_a,
    json_path_b,
    noise_ceiling_csv,  # <--- New Argument
    save_dir,
    label_a="Result A",
    label_b="Result B",
    filename="roi_rostral_caudal_normalized_lineplot.png",
):
    # --- Define Explicit Anatomical Order (Rostral -> Caudal) ---
    anatomical_order = [
        # --- ROSTRAL (FRONT) ---
        "10pp", "a10p", "p10p", "10r", "10d", "10v",
        # Orbital / Ventral
        "11l", "a47r", "47s", "47m", "47l", "p47r", "13l",
        # Dorsolateral Anterior
        "9a", "46", "a9-46v",
        # Dorsolateral Mid/Posterior
        "9-46d", "p9-46v", "9m", "9p",
        # Inferior Frontal Gyrus
        "IFSa", "45", "IFSp", "44",
        # Posterior / Superior / Junction
        "8Av", "8Ad", "8C", "8BL", "8BM",
        "IFJa", "IFJp",
        "SFL", "s6-8", "i6-8",
        # --- CAUDAL (BACK) ---
    ]

    # 1) Load Data
    print(f"Loading Results A: {os.path.basename(json_path_a)}")
    a_raw = load_max_r_by_roi(json_path_a, anatomical_order)
    
    print(f"Loading Results B: {os.path.basename(json_path_b)}")
    b_raw = load_max_r_by_roi(json_path_b, anatomical_order)

    print(f"Loading Noise Ceilings: {os.path.basename(noise_ceiling_csv)}")
    ceilings = load_noise_ceilings(noise_ceiling_csv)

    # 2) Filter ROIs: Must exist in A, B, AND have a valid noise ceiling
    valid_rois = []
    norm_vals_a = []
    norm_vals_b = []

    for roi in anatomical_order:
        # Check if ROI exists in both result sets
        if roi not in a_raw or roi not in b_raw:
            continue
        
        # Check if ROI exists in noise ceilings
        if roi not in ceilings:
            print(f"Skipping {roi}: No noise ceiling found.")
            continue

        nc = ceilings[roi]

        # Avoid division by zero or extremely low noise ceilings
        if nc <= 0.01:
            print(f"Skipping {roi}: Noise ceiling too low or zero ({nc:.4f}).")
            continue

        # Normalize
        norm_a = a_raw[roi] / nc
        norm_b = b_raw[roi] / nc

        valid_rois.append(roi)
        norm_vals_a.append(norm_a)
        norm_vals_b.append(norm_b)

    if not valid_rois:
        raise ValueError("No valid ROIs found (overlap of A, B, and Noise Ceilings > 0).")

    print(f"Plotting {len(valid_rois)} ROIs.")

    # 3) Plot
    plt.figure(figsize=(14, 7))
    x = np.arange(len(valid_rois))

    plt.plot(x, norm_vals_a, marker="o", linewidth=2.5, label=label_a, color="#2c7bb6")  # blue
    plt.plot(x, norm_vals_b, marker="o", linewidth=2.5, label=label_b, color="#d7191c")  # red

    # Add a reference line at 1.0 (Ceiling)
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Noise Ceiling")

    plt.xlabel("Prefrontal ROIs (Ordered Rostral → Caudal)", fontsize=12, fontweight="bold")
    plt.ylabel("Normalized Pearson's r (r / noise_ceiling)", fontsize=12, fontweight="bold")
    plt.title("Noise-Normalized Decoding Performance: Rostral to Caudal Gradient", fontsize=14)

    plt.xticks(x, valid_rois, rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")

    plt.show()

if __name__ == "__main__":
    # Paths
    results_a = "/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay/results.json"
    results_b = "/mnt/store1/lucas/checkpoints/tf_reg/untrained/tf_medium_untrained/results/frame-only_enc+delay_delay_lsa_wfdelay/results.json"
    
    # Path to your TSV/CSV with noise ceilings
    noise_csv = "/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lsa_wfdelay/64kDense/sub-01/_reliability_roi/split_half_roi_reliability_fisherz.tsv"

    save_dir = "/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay"

    plot_roi_comparison(
        results_a,
        results_b,
        noise_csv,
        save_dir,
        label_a="Trained",
        label_b="Untrained",
    )