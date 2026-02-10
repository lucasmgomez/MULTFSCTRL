import os
import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib
import argparse
import ast

# ---------------------------------------------------------
# 1. Helpers 
# ---------------------------------------------------------

TASKDIR_RE = re.compile(r"^task-(?P<task>[^_]+)_acq-(?P<acq>.+)_run-(?P<run>\d+)$")

def parse_taskdir_name(taskdir_basename: str):
    m = TASKDIR_RE.match(taskdir_basename)
    if not m:
        return None
    return m.group("task"), m.group("acq"), m.group("run")

def tc_format(task, tc):
    """
    Formats the task condition string based on specific task rules.
    Matches the logic in the provided build_data reference.
    """
    task_tc_len_map = {'ctxdm': 6, 'interdms': 8, '1back': 12} 

    correct_len = task_tc_len_map.get(task, len(str(tc))) 
    if len(str(tc)) < correct_len:
        tc = tc.zfill(correct_len)   

    if task == '1back': # truncates to 5 stimuli (10 loc x obj pairs)
        tc = tc[:10]

    return tc         

def get_trial_betas(betas_task_dir, base_events, trial_num):
    """
    Loads all beta files corresponding to a specific trial number.
    Returns: numpy array of shape (n_events, n_vertices)
    """
    trial_str = f"Trial{trial_num:02d}"
    trial_events = base_events[base_events['trial'] == trial_str]
    
    if trial_events.empty:
        raise ValueError(f"No events found for {trial_str}")

    eids = trial_events['event_id'].to_list()
    loaded_betas = []

    for eid in eids:
        # Glob for the specific event dscalar
        beta_file_pattern = os.path.join(betas_task_dir, f"*{eid}*.dscalar.nii")
        matches = glob.glob(beta_file_pattern)
        
        if not matches:
             raise FileNotFoundError(f"No beta file found for pattern: {beta_file_pattern}")
        
        # Load and squeeze to (N_vertices,)
        b = nib.load(matches[0]).get_fdata().squeeze().astype(np.float32)
        loaded_betas.append(b)

    if not loaded_betas:
        raise ValueError("No betas loaded.")

    # Stack along time/event axis: (n_events, n_vertices)
    return np.stack(loaded_betas, axis=0)

# ---------------------------------------------------------
# 2. ROI Masking Helpers
# ---------------------------------------------------------

def load_atlas_data(dlabel_path):
    dl = nib.load(dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    label_dict = dl.header.get_axis(0).label[0] 
    return label_dict, data

def create_roi_mask(dlabel_info, roi_name, lateralize='LR'):
    label_dict, data = dlabel_info
    target_labels = []
    
    if lateralize == 'LR':
        target_labels = [f'L_{roi_name}_ROI', f'R_{roi_name}_ROI']
    elif lateralize == 'L':
        target_labels = [f'L_{roi_name}_ROI']
    elif lateralize == 'R':
        target_labels = [f'R_{roi_name}_ROI']
    else:
        target_labels = [roi_name]

    matched_keys = [k for k, (name, _) in label_dict.items() if name in target_labels]
    mask = np.isin(data, matched_keys)
    return mask

# ---------------------------------------------------------
# 3. Main Processing Logic
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute ROI Averaged Betas by Task Condition (TC)")

    # Paths
    parser.add_argument("--behav_dir", type=str, default="/mnt/tempdata/lucas/fmri/recordings/TR/behav")
    parser.add_argument("--betas_dir", type=str, default="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lss/64kDense")
    parser.add_argument("--dlabel_path", type=str, default="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii")
    parser.add_argument("--save_path", type=str, default="./roi_results")
    
    # Config
    parser.add_argument("--subj", type=str, default="sub-01")
    parser.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02", "ses-03", "ses-04"])
    parser.add_argument("--events_type", type=str, default='base')
    parser.add_argument("--lateralize", type=str, default='LR', choices=['LR', 'L', 'R'])
    
    default_rois = [
        'SFL', 'i6-8', 's6-8', 'IFJa', 'IFJp', 'IFSp', 'IFSa', '8BM', '8Av', '8Ad', 
        '8BL', '8C', '9m', '9p', '9a', '9-46d', 'a9-46v', 'p9-46v', '46', '44', 
        '45', '47l', '47m', '47s', 'a47r', 'p47r', '10r', '10d', '10v', 'a10p', 
        'p10p', '10pp', '11l', '13l'
    ]
    parser.add_argument("--rois", type=str, default=str(default_rois))

    args = parser.parse_args()

    # --- 1. Load Atlas & ROIs ---
    try:
        roi_list = ast.literal_eval(args.rois)
    except Exception:
        print("Error parsing ROIs. Using default list.")
        roi_list = default_rois

    print(f"Loading Atlas: {args.dlabel_path}")
    atlas_info = load_atlas_data(args.dlabel_path)
    
    print("Pre-calculating ROI masks...")
    roi_masks = {}
    for roi in roi_list:
        mask = create_roi_mask(atlas_info, roi, args.lateralize)
        if mask.sum() > 0:
            roi_masks[roi] = mask

    # --- 2. Build Data Structure (Stacking Repeats Across Sessions) ---
    # Structure: betas[folder_name][tc] = [list of (n_events, n_vertices) arrays]
    betas_storage = {}
    
    subj_betas_dir = os.path.join(args.betas_dir, args.subj)
    subj_behav_dir = os.path.join(args.behav_dir, args.subj)

    for session in args.sessions:
        print(f"Scanning Session: {session}...")
        betas_ses_dir = os.path.join(subj_betas_dir, session)
        behav_ses_dir = os.path.join(subj_behav_dir, session)

        if not os.path.exists(betas_ses_dir):
            print(f"  Warning: Directory not found: {betas_ses_dir}")
            continue

        files = [d for d in os.listdir(betas_ses_dir) if d.startswith("task-")]

        for f in files:
            # Initialize storage for this run folder if not exists
            if f not in betas_storage:
                betas_storage[f] = {}

            parsed = parse_taskdir_name(f)
            if not parsed: continue
            task, acq, run = parsed
            
            task_tag = f"task-{task}"

            # Locate Block Log
            hit = glob.glob(os.path.join(behav_ses_dir, f"*{task_tag}_{acq}*block_{int(run)-1}*scored*.tsv"))
            if not hit:
                continue
            
            block_df = pd.read_csv(hit[0], sep="\t")
            block_df['TrialNumber'] = block_df['TrialNumber'].astype(int)
            block_df['tc'] = block_df['tc'].astype(int)

            # Locate Events File
            events_tsv = os.path.join(behav_ses_dir, f"events_{args.events_type}/{task_tag}_{acq}_run-{run}_base-events.tsv")
            if not os.path.exists(events_tsv):
                 print(f"  Skipping missing events file: {events_tsv}")
                 continue

            events = pd.read_csv(events_tsv, sep="\t")
            events = events.sort_values(by=['onset']).reset_index(drop=True)
            
            betas_task_dir = os.path.join(betas_ses_dir, f)

            # Process Trials
            for trial, raw_tc in zip(block_df['TrialNumber'], block_df['tc']):
                tc = tc_format(task, str(raw_tc))
                
                try:
                    trial_betas = get_trial_betas(betas_task_dir, events, trial)
                    
                    if tc not in betas_storage[f]:
                        betas_storage[f][tc] = [trial_betas]
                    else:
                        betas_storage[f][tc].append(trial_betas)
                        
                except Exception as e:
                    # print(f"Error processing {f} trial {trial}: {e}")
                    continue

    # --- 3. Compute Averages and Extract ROI values ---
    print("\nComputing averages and extracting ROI data...")
    results_data = []

    for f_name, tc_dict in betas_storage.items():
        parsed = parse_taskdir_name(f_name)
        if not parsed: continue
        task, acq, run = parsed

        for tc, trial_list in tc_dict.items():
            if not trial_list:
                continue

            # 1. Stack Repeats: (n_repeats, n_events, n_vertices)
            stacked_repeats = np.stack(trial_list, axis=0)
            
            # 2. Average over Session Repeats: (n_events, n_vertices)
            # This matches: np.mean(np.stack(betas[f][tc], axis=0), axis=0) from your reference
            mean_repeats = np.mean(stacked_repeats, axis=0)
            
            # 3. Average over Time (Events): (n_vertices,)
            final_map = np.mean(mean_repeats, axis=0)

            # 4. Extract ROI means
            for roi_name, mask in roi_masks.items():
                val = np.mean(final_map[mask])
                
                results_data.append({
                    "subject": args.subj,
                    "task": task,
                    "acq": acq,
                    "run": run,
                    "tc": tc,
                    "roi": roi_name,
                    "mean_beta": val,
                    "n_repeats": len(trial_list)
                })

    # --- 4. Save CSV ---
    if results_data:
        df = pd.DataFrame(results_data)
        
        # Sort for cleanliness
        df = df.sort_values(by=["task", "acq", "run", "tc", "roi"])
        
        os.makedirs(args.save_path, exist_ok=True)
        out_csv = os.path.join(args.save_path, f"{args.subj}_roi_averages_by_tc.csv")
        
        df.to_csv(out_csv, index=False)
        print(f"Done! Saved results to: {out_csv}")
        print(df.head())
    else:
        print("No data processed. Check paths and logs.")

if __name__ == "__main__":
    main()