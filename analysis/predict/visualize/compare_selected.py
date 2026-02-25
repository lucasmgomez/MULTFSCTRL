import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. ROI Masking Setup (Updated with your functions)
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
# 2. Main Extraction & Z-Scoring
# ---------------------------------------------------------

# ---------------------------------------------------------
# 2. Main Extraction & Z-Scoring
# ---------------------------------------------------------
def extract_session5_roi_betas():
    dlabel_path = "/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii"
    base_betas_dir = "/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/ctrl_run/glm_runs/lsa/64kDense/sub-01/ses-05"
    base_events_dir = "/mnt/tempdata/lucas/fmri/recordings/TR/behav/sub-01/ses-05/events_wfdelay"
    base_block_dir = "/home/lucas/projects/task_stimuli/data/multfs/trevor/blockfiles/session05_rep"
    
    rois = ['8BL', '8C', '9-46d', 'p9-46v', '10v', '46']
    
    print("Loading Glasser Atlas...")
    dlabel_info = load_atlas_data(dlabel_path)
    roi_masks = {roi: create_roi_mask(dlabel_info, roi) for roi in rois}
    
    roi_data = {roi: {'betas': [], 'trial_type': []} for roi in rois}

    event_files = glob.glob(os.path.join(base_events_dir, "*_base-events.tsv"))
    print(f"Found {len(event_files)} event files. Extracting betas...")

    import re # Just in case it's not at the top of your file
    
    for event_file in event_files:
        basename = os.path.basename(event_file)
        
        # Regex extracts group 1 (task), group 2 (acq), and group 3 (run digits)
        match = re.search(r'task-([^_]+)_(.+)_run-(\d+)_base-events\.tsv', basename)
        if not match:
            print(f"Warning: Could not parse file name structure for {basename}")
            continue
            
        task = match.group(1)
        acq = match.group(2)  # This safely captures 'a_lo', 'lol', or 'ctg_ABBCCA'
        run_str = match.group(3)
        run_formatted = f"run-{run_str}"
        
        # Convert run digits to block integer (e.g., '01' -> block 0)
        run_idx = int(run_str) - 1
        
        # Now we just plug 'acq' directly into the blockfile name
        block_file = os.path.join(base_block_dir, f"{task}_{acq}_block_{run_idx}.csv")
        if not os.path.exists(block_file):
            print(f"Skipping: Block file not found -> {block_file}")
            continue
            
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
            
            beta_search = glob.glob(os.path.join(task_beta_dir, f"*{event_id}_beta.dscalar.nii"))
            if not beta_search: continue
            
            try:
                beta_data = nib.load(beta_search[0]).get_fdata() 
                for roi in rois:
                    roi_betas = beta_data[:, roi_masks[roi]] 
                    avg_roi_beta = np.mean(roi_betas, axis=1, keepdims=True) # Keeps shape as (1, 1)
                    roi_data[roi]['betas'].append(avg_roi_beta)
                    roi_data[roi]['trial_type'].append(t_type)
            except Exception as e:
                print(f"Error loading {beta_search[0]}: {e}")
                continue

    # Z-Score and Split per ROI
    print("\nCalculating Vertex-wise Z-scores and isolating ROI-specific targets...")
    results = {}

    for roi in rois:
        if not roi_data[roi]['betas']: continue

        betas_arr = np.concatenate(roi_data[roi]['betas'], axis=0)
        types_arr = np.array(roi_data[roi]['trial_type'])
        
        mean_val = np.mean(betas_arr, axis=0)
        std_val = np.std(betas_arr, axis=0)
        std_val[std_val == 0] = 1e-8 
        z_betas = (betas_arr - mean_val) / std_val
        
        target_outlier_label = f"outlier_{roi}"
        random_mask = (types_arr == 'random')
        selected_mask = (types_arr == target_outlier_label)
        
        results[roi] = {
            'random': z_betas[random_mask],       
            'selected': z_betas[selected_mask]    
        }
        
        print(f"ROI: {roi:^8} | Vertices: {betas_arr.shape[1]:<4} | "
              f"Random Trials: {len(results[roi]['random']):<3} | Selected Target Trials: {len(results[roi]['selected']):<3}")

    return results

# ---------------------------------------------------------
# 3. Plotting Distributions
# ---------------------------------------------------------
def plot_roi_distributions(results):
    rois = list(results.keys())
    
    # Create a 3x4 grid: 3 rows, 4 columns (2 columns per ROI: 1 for Dist, 1 for Bar)
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(22, 16))
    sns.set_theme(style="whitegrid")
    
    for i, roi in enumerate(rois):
        # Calculate grid position for this ROI's two plots
        row = i // 2
        col_base = (i % 2) * 2
        ax_dist = axes[row, col_base]       # Left plot: Distribution
        ax_bar = axes[row, col_base + 1]    # Right plot: Bar plot
        
        random_betas = results[roi]['random'].flatten()
        selected_betas = results[roi]['selected'].flatten()
        
        if len(random_betas) == 0 or len(selected_betas) == 0:
            ax_dist.set_title(f"{roi} (Insufficient Data)")
            ax_dist.axis('off')
            ax_bar.axis('off')
            continue
            
        # ==========================================
        # 1. KDE Distribution Plot (Left)
        # ==========================================
        sns.kdeplot(x=random_betas, ax=ax_dist, color='#3498db', label='Random Density', linewidth=2, fill=True, alpha=0.1)
        sns.kdeplot(x=selected_betas, ax=ax_dist, color='#e74c3c', label='Targeted Density', linewidth=2, fill=True, alpha=0.1)
        
        rand_mean, rand_med = np.mean(random_betas), np.median(random_betas)
        sel_mean, sel_med = np.mean(selected_betas), np.median(selected_betas)
        
        ax_dist.axvline(rand_mean, color='#2980b9', linestyle='-', linewidth=2, label='Random Mean')
        ax_dist.axvline(rand_med, color='#2980b9', linestyle='--', linewidth=2, label='Random Median')
        ax_dist.axvline(sel_mean, color='#c0392b', linestyle='-', linewidth=2, label='Targeted Mean')
        ax_dist.axvline(sel_med, color='#c0392b', linestyle='--', linewidth=2, label='Targeted Median')
        
        ax_dist.set_title(f"{roi}: Distribution", fontsize=14, fontweight='bold')
        ax_dist.set_xlabel("Average ROI Z-Scored Beta")
        ax_dist.set_ylabel("Density")
        ax_dist.legend(fontsize=9, loc='upper right')
        
        p1, p99 = np.percentile(np.concatenate([random_betas, selected_betas]), [1, 99])
        ax_dist.set_xlim(p1 - 1, p99 + 1)

        # ==========================================
        # 2. Mean Bar Plot with SEM Error Bars (Right)
        # ==========================================
        # Calculate Standard Error of the Mean (SEM) for the error bars
        rand_sem = np.std(random_betas, ddof=1) / np.sqrt(len(random_betas))
        sel_sem = np.std(selected_betas, ddof=1) / np.sqrt(len(selected_betas))
        
        # Plot the bars
        bars = ax_bar.bar(
            x=['Random', f'Targeted\n({roi})'], 
            height=[rand_mean, sel_mean], 
            yerr=[rand_sem, sel_sem], 
            capsize=8,              # Horizontal caps on the error bars
            color=['#3498db', '#e74c3c'], 
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5
        )
        
        ax_bar.set_title(f"{roi}: Mean Activation", fontsize=14, fontweight='bold')
        ax_bar.set_ylabel("Mean Z-Scored Beta")
        
        # Add a faint horizontal line at y=0 for reference
        ax_bar.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)

    plt.suptitle("Session 5 Delay Betas: Random vs Targeted Outlier Trials", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("./roi_beta_distributions.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    roi_comparison_results = extract_session5_roi_betas()
    if roi_comparison_results:
        plot_roi_distributions(roi_comparison_results)