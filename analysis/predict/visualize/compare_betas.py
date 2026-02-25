import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import os
import pandas as pd
import re
import numpy as np
import glob
import nibabel as nib
import torch

# Regex to parse task info from beta file names
TASKDIR_RE = re.compile(r"^task-(?P<task>[^_]+)_acq-(?P<acq>.+)_run-(?P<run>\d+)$")

def parse_taskdir_name(taskdir_basename: str):
    m = TASKDIR_RE.match(taskdir_basename)
    if not m:
        return None
    return m.group("task"), m.group("acq"), m.group("run")

def get_betas(betas_task_dir, base_events, trial_num):
    trial_events = base_events[base_events['trial'] == f"Trial{trial_num:02d}"]
    eids = trial_events['event_id'].to_list()

    betas = []
    for eid in eids:
        beta_file_pattern = os.path.join(betas_task_dir, f"*{eid}*.dscalar.nii")
        matches = glob.glob(beta_file_pattern)
        if not matches:
             raise FileNotFoundError(f"No beta file found for pattern: {beta_file_pattern}")
        beta_file = matches[0]
        b = nib.load(beta_file).get_fdata().squeeze().astype(np.float32)
        betas.append(b)

    return np.stack(betas, axis=0)


def tc_format(task, tc):
    task_tc_len_map = {'ctxdm': 6, 'interdms': 8, '1back': 12} 

    correct_len = task_tc_len_map.get(task, len(str(tc))) 
    if len(str(tc)) < correct_len:
        tc = tc.zfill(correct_len)   

    if task == '1back': # truncates to 5 stimuli (10 loc x obj pairs)
        tc = tc[:10]

    return tc         

def build_data(behav_dir, betas_dir, subj, sessions, events_type='wfdelay'):
    betas = {}

    for session in sessions:
        behav_ses_dir = os.path.join(behav_dir, subj, session)
        betas_ses_dir = os.path.join(betas_dir, subj, session)

        if not os.path.exists(betas_ses_dir):
            print(f"Warning: Directory not found: {betas_ses_dir}")
            continue

        files = [d for d in os.listdir(betas_ses_dir) if d.startswith("task-")]

        for f in files:
            betas[f] = {}

            parsed = parse_taskdir_name(f)
            if not parsed: continue
            task, acq, run = parsed
            
            task_tag = f"task-{task}"

            hit = glob.glob(os.path.join(behav_ses_dir, f"*{task_tag}_{acq}*block_{int(run)-1}*scored*.tsv"))

            if not hit:
                # print(f"Skipping missing block log for {subj} {session} {task_tag} run-{run}")
                continue
        
            block_log_tsv = hit[0]
            block_df = pd.read_csv(block_log_tsv, sep="\t")
            block_df['TrialNumber'] = block_df['TrialNumber'].astype(int)
            block_df['tc'] = block_df['tc'].astype(int)

            events_tsv = os.path.join(behav_ses_dir, f"events_{events_type}/{task_tag}_{acq}_run-{run}_base-events.tsv")
            if not os.path.exists(events_tsv):
                 print(f"Skipping missing events file: {events_tsv}")
                 continue

            events = pd.read_csv(events_tsv, sep="\t")
            events = events.sort_values(by=['onset']).reset_index(drop=True)
                                      
            betas_task_dir = os.path.join(betas_ses_dir, f"{task_tag}_acq-{acq}_run-{run}")

            for trial, tc in zip(block_df['TrialNumber'], block_df['tc']):
                tc = tc_format(task, str(tc))
                
                # Collect betas
                try:
                    trial_betas = get_betas(betas_task_dir, events, trial)
                    if tc not in betas[f]:
                        betas[f][tc] = [trial_betas]
                    else:
                        betas[f][tc].append(trial_betas)

                except Exception as e:
                    print(f"Error processing {f} trial {trial}: {e}")
                    continue

    # Average betas over session repeats
    for f in betas.keys():
        for tc in list(betas[f].keys()): 
            if len(betas[f][tc]) > 0:
                betas[f][tc] = np.mean(np.stack(betas[f][tc], axis=0), axis=0)
    
    return betas

def aggregate_betas(betas_dict):
    """
    Flattens the nested betas dictionary into a single 2D numpy array.
    Output shape: (total_trials, num_voxels)
    """
    all_trials = []
    for f in betas_dict.values():
        for tc_data in f.values():
            # If tc_data is a single 1D array, expand it so we can concatenate
            if tc_data.ndim == 1:
                tc_data = np.expand_dims(tc_data, axis=0)
            all_trials.append(tc_data)
            
    if not all_trials:
        return np.array([])
        
    return np.concatenate(all_trials, axis=0)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def compare_sessions_combined_z(behav_dir, betas_dir, subj):
    print("Loading Sessions 1-4...")
    sessions_1_4 = ['ses-01', 'ses-02', 'ses-03', 'ses-04']
    betas_dict_1_4, _ = build_data(behav_dir, betas_dir, subj, sessions_1_4)
    
    print("Loading Session 5...")
    sessions_5 = ['ses-05']
    betas_dict_5, _ = build_data(behav_dir, betas_dir, subj, sessions_5)
    
    # Aggregate into (Trials, Voxels) matrices
    b_1_4 = aggregate_betas(betas_dict_1_4)
    b_5 = aggregate_betas(betas_dict_5)
    
    if b_1_4.size == 0 or b_5.size == 0:
        raise ValueError("Insufficient data loaded. Check your directories.")

    n_trials_1_4 = b_1_4.shape[0]
    
    # ---------------------------------------------------------
    # 1. Combine, Z-Score, and Split
    # ---------------------------------------------------------
    print("\nConcatenating and Z-scoring all sessions together...")
    b_all = np.concatenate((b_1_4, b_5), axis=0)
    
    # Calculate global mean and std across all 5 sessions
    mean_all = np.mean(b_all, axis=0)
    std_all = np.std(b_all, axis=0)
    std_all[std_all == 0] = 1e-8 # Prevent division by zero on flat voxels
    
    # Z-score the entire dataset
    z_all = (b_all - mean_all) / std_all
    
    # Split them back apart using the row index
    z_1_4 = z_all[:n_trials_1_4, :]
    z_5   = z_all[n_trials_1_4:, :]

    print(f"Shape of Z-scored Sessions 1-4: {z_1_4.shape}")
    print(f"Shape of Z-scored Session 5: {z_5.shape}")

    # ---------------------------------------------------------
    # 2. Compare the Z-Scored Distributions
    # ---------------------------------------------------------
    print("Generating distribution plots...")
    plt.figure(figsize=(12, 5))

    # Plot A: Global Voxel Mean Distribution
    plt.subplot(1, 2, 1)
    sns.kdeplot(np.mean(z_1_4, axis=0), fill=True, label='Sessions 1-4 (Z-scored)')
    sns.kdeplot(np.mean(z_5, axis=0), fill=True, label='Session 5 (Z-scored)')
    plt.title("Distribution of Voxel Means (Global Z)")
    plt.xlabel("Mean Z-Score")
    plt.ylabel("Density")
    plt.legend()

    # Plot B: Voxel-wise Correlation
    plt.subplot(1, 2, 2)
    mean_z_1_4 = np.mean(z_1_4, axis=0)
    mean_z_5 = np.mean(z_5, axis=0)
    
    plt.scatter(mean_z_1_4, mean_z_5, alpha=0.1, s=1)
    
    # Calculate Pearson correlation across voxels
    r, p = stats.pearsonr(mean_z_1_4, mean_z_5)
    
    plt.title(f"Voxel-wise Mean Correlation (r={r:.3f})")
    plt.xlabel("Sessions 1-4 Mean Z-Score")
    plt.ylabel("Session 5 Mean Z-Score")
    
    # Identity line
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  
        np.max([plt.xlim(), plt.ylim()]),  
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    
    plt.tight_layout()
    plt.show()

    return z_1_4, z_5
# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    # Replace these with your actual paths and subject ID
    BEHAV_DIR = "/mnt/tempdata/lucas/fmri/recordings/TR/behav"
    BETAS_DIR = "/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/ctrl_run/glm_runs/lsa/64kDense"
    SUBJECT   = "sub-01"
    
    try:
        z_scores, voxel_deviations = compare_sessions_combined_z(
            behav_dir=BEHAV_DIR, 
            betas_dir=BETAS_DIR, 
            subj=SUBJECT
        )
    except Exception as e:
        print(f"Execution failed: {e}")