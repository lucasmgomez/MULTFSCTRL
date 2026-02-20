import numpy as np
import os
import argparse
import ast
import pandas as pd
import sys
import torch 

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_conditions_from_pth(acts_fp, selected_indices):
    """
    Loads the .pth file and retrieves the trial conditions ('tcs') 
    for the specific indices provided.
    """

    # Load dictionary (CPU is safer for generic scripts)
    acts_dict = torch.load(acts_fp, map_location='cpu') 

    all_conditions = acts_dict['tcs']
    
    # Extract specific conditions based on the outlier indices
    # We assume 1-to-1 mapping between beta rows and condition list indices
    selected_conds = []
    for idx in selected_indices:
        if idx < len(all_conditions):
            selected_conds.append(all_conditions[idx])
        else:
            selected_conds.append("INDEX_OUT_OF_BOUNDS")
            
    return selected_conds

def tc_format(task, tc):
    task_tc_len_map = {
        "InterDMS": 12,
        "Oneback": 12,
        "Twoback": 12,
        "ctxDM": 6,
        "DMSO": 4,
        "DMSA": 4
    }

    correct_len = task_tc_len_map.get(task, len(str(tc))) 
    if len(str(tc)) < correct_len:
        tc = tc.zfill(correct_len)   

    return tc         


def main():
    parser = argparse.ArgumentParser(description="Sample Outlier Trials -> Master CSV")
    parser.add_argument("--lateralization", type=str, default="LH_RH")
    
    # Example: "['task-DMSO_LO', 'task-ctxDM_LOL']"
    parser.add_argument("--ctrl_tasks", type=str, required=True, 
                        help="List of tasks to analyze.")
    
    parser.add_argument("--rois", type=str, default="['10r', '46']")
    parser.add_argument("--base_dir", type=str, default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed")
    
    parser.add_argument("--decode_results_dir", type=str, 
                        default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay",
                        help="Path containing the 'betas.npz' file")
    
    parser.add_argument("--run_name", type=str, default="128c")
    parser.add_argument("--n_percentile", type=int, default=95, help="Percentile threshold.")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of trials to sample per ROI/Task.")

    args = parser.parse_args()
    
    # Parse inputs
    try:
        rois = ast.literal_eval(args.rois)
        ctrl_tasks_list = ast.literal_eval(args.ctrl_tasks)
    except Exception as e:
        print(f"Error parsing lists: {e}")
        sys.exit(1)
        
    results_dir = os.path.join(args.base_dir, "results")
    activations_base_dir = os.path.join(args.base_dir, "activations") 
    
    # ---------------------------------------------------------
    # 1. Load True Betas
    # ---------------------------------------------------------
    true_betas_path = f"{args.decode_results_dir}/betas.npz"
    print(f"Loading Global True Betas from: {true_betas_path}")
    
    if not os.path.exists(true_betas_path):
        raise FileNotFoundError(f"True betas file not found: {true_betas_path}")
    
    true_betas_dict = np.load(true_betas_path)

    # ---------------------------------------------------------
    # 2. Main Processing Loop
    # ---------------------------------------------------------
    print(f"\nScanning {len(ctrl_tasks_list)} tasks...")
    
    # Accumulate all selected rows here
    master_data_rows = []

    for task_name in ctrl_tasks_list:
        print(f"--- Processing: {task_name} ---")
        
        # A. Load Predictions
        pred_betas_path = os.path.join(results_dir, f"ctrl_pred_betas/{args.lateralization}/{args.run_name}/{task_name}/results.npz")
        
        if not os.path.exists(pred_betas_path):
            print(f"  [Skip] Prediction file missing.")
            continue
            
        try:
            pred_betas = np.load(pred_betas_path)
        except Exception as e:
            print(f"  [Skip] Error loading predictions: {e}")
            continue

        # B. Define Activation Path for this Task (Loaded only if needed)
        acts_path = os.path.join(activations_base_dir, f"{task_name}.pth")

        # C. Process Each ROI
        for region in rois:
            if region not in true_betas_dict or region not in pred_betas:
                continue 
            
            # 1. Calculate Threshold
            roi_tb = true_betas_dict[region].flatten()
            threshold_val = np.percentile(roi_tb, args.n_percentile)
            
            # 2. Find Outliers
            roi_pb_trials = pred_betas[region]
            if roi_pb_trials.ndim == 1: roi_pb_trials = roi_pb_trials.reshape(-1, 1)
            
            high_activation_mask = np.any(roi_pb_trials >= threshold_val, axis=1)
            outlier_indices = np.where(high_activation_mask)[0]
            
            count = len(outlier_indices)
            
            # 3. Check Condition: Must have at least N samples
            if count < args.n_samples:
                continue
            
            # 4. Random Sampling
            # np.random.seed(42) # Optional: uncomment for deterministic sampling
            selected_indices = np.random.choice(outlier_indices, size=args.n_samples, replace=False)
            selected_indices = np.sort(selected_indices)

            # 5. Load Conditions from .pth
            # Note: We load the .pth once per task/ROI pair. 
            # Optimization: If the .pth is huge, you might want to load it once per task outside the ROI loop,
            # but usually, this is acceptable.
            conditions = get_conditions_from_pth(acts_path, selected_indices)
            
            if conditions is None:
                continue 

            # 6. Append to Master List
            for idx, cond in zip(selected_indices, conditions):
                master_data_rows.append({
                    'Task': task_name,
                    'ROI': region,
                    'Trial_Index': int(idx),
                    'Condition': tc_format(task_name,cond),
                    'Threshold_Used': threshold_val
                })
            
            print(f"  [ROI {region}] Added {args.n_samples} trials.")

    # ---------------------------------------------------------
    # 3. Save Master CSV
    # ---------------------------------------------------------
    output_dir = os.path.join(results_dir, f"ctrl_pred_betas/{args.lateralization}/{args.run_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "MASTER_selected_conditions.csv")
    
    if master_data_rows:
        df = pd.DataFrame(master_data_rows)
        # Reorder for cleanliness
        df = df[['Task', 'ROI', 'Trial_Index', 'Condition', 'Threshold_Used']]
        
        df.to_csv(csv_path, index=False)
        print(f"\n[SUCCESS] Master CSV saved to:\n{csv_path}")
        print(f"Total Selected Trials: {len(df)}")
    else:
        print("\n[INFO] No trials met the criteria for any task/ROI.")

if __name__ == "__main__":
    main()