import os
import glob
import re
import json
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. ROI Masking Setup
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

# ---------------------------------------------------------
# 2. Main Extraction & Grouping (Max Delay per Trial)
# ---------------------------------------------------------
def extract_session5_roi_betas(ignored_tasks=None):
    if ignored_tasks is None:
        ignored_tasks = []
        
    dlabel_path = "/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii"
    base_betas_dir = "/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/ctrl_run/glm_runs/lsa/64kDense/sub-01/ses-05"
    base_events_dir = "/mnt/tempdata/lucas/fmri/recordings/TR/behav/sub-01/ses-05/events_wfdelay"
    base_block_dir = "/home/lucas/projects/task_stimuli/data/multfs/trevor/blockfiles/session05_rep"
    
    rois = ['8BL', '8C', '9-46d', 'p9-46v', '10v', '46']
    
    print("Loading Glasser Atlas...")
    dlabel_info = load_atlas_data(dlabel_path)
    roi_masks = {roi: create_roi_mask(dlabel_info, roi) for roi in rois}
    
    roi_data = {roi: {'trial_id': [], 'trial_type': [], 'beta_val': []} for roi in rois}

    event_files = glob.glob(os.path.join(base_events_dir, "*_base-events.tsv"))
    print(f"Found {len(event_files)} event files. Extracting betas...")

    for event_file in event_files:
        basename = os.path.basename(event_file)
        match = re.search(r'task-([^_]+)_(.+)_run-(\d+)_base-events\.tsv', basename)
        if not match: continue
            
        task = match.group(1)
        if task in ignored_tasks: continue
            
        acq = match.group(2)
        run_str = match.group(3)
        run_formatted = f"run-{run_str}"
        run_idx = int(run_str) - 1
        
        block_file = os.path.join(base_block_dir, f"{task}_{acq}_block_{run_idx}.csv")
        if not os.path.exists(block_file): continue
            
        block_df = pd.read_csv(block_file)
        block_df['trial'] = [f"Trial{i+1:02d}" for i in range(len(block_df))]
        
        events_df = pd.read_csv(event_file, sep='\t')
        delay_events = events_df[events_df['phase'] == 'Delay'].copy()
        
        if 'trial_type' in delay_events.columns:
            delay_events = delay_events.drop(columns=['trial_type'])
            
        delay_events = delay_events.merge(block_df[['trial', 'trial_type']], on='trial', how='left')

        task_beta_dir = os.path.join(base_betas_dir, f"task-{task}_acq-{acq}_{run_formatted}")
        
        for _, row in delay_events.iterrows():
            event_id = row['event_id']
            t_type = row['trial_type']
            trial_num = row['trial']
            unique_trial_id = f"{task}_{acq}_{run_formatted}_{trial_num}"
            
            beta_search = glob.glob(os.path.join(task_beta_dir, f"*{event_id}_beta.dscalar.nii"))
            if not beta_search: continue
            
            try:
                beta_data = nib.load(beta_search[0]).get_fdata() 
                for roi in rois:
                    roi_mean = np.mean(beta_data[:, roi_masks[roi]])
                    roi_data[roi]['beta_val'].append(roi_mean)
                    roi_data[roi]['trial_id'].append(unique_trial_id)
                    roi_data[roi]['trial_type'].append(t_type)
            except Exception as e:
                print(f"Error loading {beta_search[0]}: {e}")
                continue

    results = {}
    for roi in rois:
        if not roi_data[roi]['beta_val']: continue
        df = pd.DataFrame(roi_data[roi])
        mean_val, std_val = df['beta_val'].mean(), df['beta_val'].std()
        if std_val == 0: std_val = 1e-8
        df['z_beta'] = (df['beta_val'] - mean_val) / std_val
        
        trial_df = df.groupby('trial_id').agg({'z_beta': 'max', 'trial_type': 'first'}).reset_index()
        
        results[roi] = {
            'random': trial_df[trial_df['trial_type'] == 'random']['z_beta'].values,
            'selected': trial_df[trial_df['trial_type'] == f"outlier_{roi}"]['z_beta'].values
        }
    return results

# ---------------------------------------------------------
# 3. Plotting Distributions & Bar Charts
# ---------------------------------------------------------
def plot_roi_distributions(results, max_r_map=None):
    rois = list(results.keys())
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(22, 16))
    sns.set_theme(style="whitegrid")
    
    for i, roi in enumerate(rois):
        row = i // 2
        col_base = (i % 2) * 2
        ax_dist = axes[row, col_base]       
        ax_bar = axes[row, col_base + 1]    
        
        random_betas = results[roi]['random']
        selected_betas = results[roi]['selected']
        
        if len(random_betas) == 0 or len(selected_betas) == 0:
            ax_dist.set_title(f"{roi} (No Data)")
            continue

        # Get max R for this ROI
        max_r_str = ""
        if max_r_map and roi in max_r_map:
            max_r_str = f" | Max r = {max_r_map[roi]:.3f}"

        # --- KDE Distribution Plot ---
        sns.kdeplot(x=random_betas, ax=ax_dist, color='#3498db', label='Random', linewidth=2, fill=True, alpha=0.1)
        sns.kdeplot(x=selected_betas, ax=ax_dist, color='#e74c3c', label='Targeted', linewidth=2, fill=True, alpha=0.1)
        
        rand_mean, sel_mean = np.mean(random_betas), np.mean(selected_betas)
        ax_dist.axvline(rand_mean, color='#2980b9', linestyle='-', linewidth=2)
        ax_dist.axvline(sel_mean, color='#c0392b', linestyle='-', linewidth=2)
        
        ax_dist.set_title(f"{roi}: Peak Delay Dist.{max_r_str}", fontsize=14, fontweight='bold')
        ax_dist.set_xlabel("Max Z-Scored Delay")
        ax_dist.legend(fontsize=9)
        
        # --- Bar Plot with SEM ---
        rand_sem = np.std(random_betas, ddof=1) / np.sqrt(len(random_betas))
        sel_sem = np.std(selected_betas, ddof=1) / np.sqrt(len(selected_betas))
        
        ax_bar.bar(x=['Random', f'Targeted\n({roi})'], height=[rand_mean, sel_mean], 
                   yerr=[rand_sem, sel_sem], capsize=8, color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')
        ax_bar.set_title(f"{roi}: Mean Peak Delay", fontsize=14, fontweight='bold')

    plt.suptitle("Session 5: Highest Delay Activation (Random vs Targeted)", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("./roi_beta_distributions_max.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # --- 1. Load prediction results and find max R per ROI ---
    results_json_path = "/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay_pls_nps5to50/results.json"
    
    tasks_to_ignore = None

    max_r_per_roi = {}
    if os.path.exists(results_json_path):
        with open(results_json_path, 'r') as f:
            pred_data = json.load(f)
            for roi, layers in pred_data.items():
                # Extract all 'r' values from layers and find the maximum
                r_values = [layer_info['r'] for layer_info in layers.values()]
                max_r_per_roi[roi] = max(r_values)
    
    # --- 2. Run extraction and plotting ---
    roi_comparison_results = extract_session5_roi_betas(ignored_tasks=tasks_to_ignore)
    if roi_comparison_results:
        plot_roi_distributions(roi_comparison_results, max_r_map=max_r_per_roi)