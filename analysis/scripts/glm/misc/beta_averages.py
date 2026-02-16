import os
import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib
import argparse
import ast
import json
from joblib import load

# ---------------------------------------------------------
# 1. Helpers 
# ---------------------------------------------------------

TASKDIR_RE = re.compile(r"^task-(?P<task>[^_]+)_acq-(?P<acq>.+)_run-(?P<run>\d+)")

def parse_taskdir_name(taskdir_basename: str):
    m = TASKDIR_RE.match(taskdir_basename)
    if not m:
        return None
    return m.group("task"), m.group("acq"), m.group("run")

def tc_format(task, tc):
    task_tc_len_map = {'ctxdm': 6, 'interdms': 8, '1back': 12} 
    correct_len = task_tc_len_map.get(task, len(str(tc))) 
    if len(str(tc)) < correct_len:
        tc = tc.zfill(correct_len)   
    if task == '1back': 
        tc = tc[:10]
    return tc         

def get_trial_betas(betas_task_dir, base_events, trial_num):
    trial_str = f"Trial{trial_num:02d}"
    trial_events = base_events[base_events['trial'] == trial_str]
    
    if trial_events.empty:
        raise ValueError(f"No events found for {trial_str}")

    eids = trial_events['event_id'].to_list()
    loaded_betas = []

    for eid in eids:
        beta_file_pattern = os.path.join(betas_task_dir, f"*{eid}*.dscalar.nii")
        matches = glob.glob(beta_file_pattern)
        
        if not matches:
             raise FileNotFoundError(f"No beta file found for pattern: {beta_file_pattern}")
        
        b = nib.load(matches[0]).get_fdata().squeeze().astype(np.float32)
        loaded_betas.append(b)

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
    parser = argparse.ArgumentParser(description="Compute ROI Averaged Betas using pre-saved Scalers")

    # Paths
    parser.add_argument("--behav_dir", type=str, default="/mnt/tempdata/lucas/fmri/recordings/TR/behav")
    parser.add_argument("--betas_dir", type=str, default="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lsa_wfdelay/64kDense")
    parser.add_argument("--dlabel_path", type=str, default="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii")
    parser.add_argument("--save_path", type=str, default="./roi_results")
    
    # New Argument for Saved Scalers
    parser.add_argument("--decode_results_dir", type=str, 
                        default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay")
    
    # Config
    parser.add_argument("--subj", type=str, default="sub-01")
    parser.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02", "ses-03", "ses-04"])
    parser.add_argument("--events_type", type=str, default='wfdelay')
    parser.add_argument("--lateralize", type=str, default='LR', choices=['LR', 'L', 'R'])
    
    default_rois = ['10pp','10v','47s','46','9-46d']
    parser.add_argument("--rois", type=str, default=str(default_rois))

    args = parser.parse_args()

    # --- 1. Load Atlas & ROIs ---
    try:
        roi_list = ast.literal_eval(args.rois)
    except Exception:
        roi_list = default_rois

    atlas_info = load_atlas_data(args.dlabel_path)
    
    roi_masks = {}
    for roi in roi_list:
        mask = create_roi_mask(atlas_info, roi, args.lateralize)
        if mask.sum() > 0:
            roi_masks[roi] = mask

    # --- 2. Build Data Structure (Betas) ---
    betas_storage = {}
    subj_betas_dir = os.path.join(args.betas_dir, args.subj)
    subj_behav_dir = os.path.join(args.behav_dir, args.subj)

    for session in args.sessions:
        print(f"Scanning Session: {session}...")
        betas_ses_dir = os.path.join(subj_betas_dir, session)
        behav_ses_dir = os.path.join(subj_behav_dir, session)

        if not os.path.exists(betas_ses_dir): continue

        files = [d for d in os.listdir(betas_ses_dir) if d.startswith("task-")]
        for f in files:
            if f not in betas_storage: betas_storage[f] = {}
            parsed = parse_taskdir_name(f)
            if not parsed: continue
            task, acq, run = parsed
            
            hit = glob.glob(os.path.join(behav_ses_dir, f"*{task}_{acq}*block_{int(run)-1}*scored*.tsv"))
            if not hit: continue
            
            block_df = pd.read_csv(hit[0], sep="\t")
            events_tsv = os.path.join(behav_ses_dir, f"events_{args.events_type}/task-{task}_{acq}_run-{run}_base-events.tsv")
            if not os.path.exists(events_tsv): continue

            events = pd.read_csv(events_tsv, sep="\t").sort_values(by=['onset']).reset_index(drop=True)
            betas_task_dir = os.path.join(betas_ses_dir, f)

            for trial, raw_tc in zip(block_df['TrialNumber'], block_df['tc']):
                tc = tc_format(task, str(raw_tc))
                try:
                    trial_betas = get_trial_betas(betas_task_dir, events, trial)
                    if '1back' in task:
                        trial_betas = trial_betas[:10, :]
                    # Select only delays
                    trial_betas = trial_betas[1::2, :]
                    if tc not in betas_storage[f]: betas_storage[f][tc] = []
                    betas_storage[f][tc].append(trial_betas)
                except Exception: continue

    # --- 3. Accumulate & Average Repeats ---
    data_accumulator = [] 
    metadata_accumulator = []

    for f_name, tc_dict in betas_storage.items():
        parsed = parse_taskdir_name(f_name)
        if not parsed: continue
        task, acq, run = parsed

        for tc, trial_list in tc_dict.items():
            if not trial_list: continue
            mean_repeats = np.mean(np.stack(trial_list, axis=0), axis=0)
            data_accumulator.append(mean_repeats)
            metadata_accumulator.append({
                "subject": args.subj, "task": task, "acq": acq, "run": run, "tc": tc,
                "n_repeats": len(trial_list), "n_events": mean_repeats.shape[0]
            })

    if not data_accumulator:
        return

    full_stack_raw = np.concatenate(data_accumulator, axis=0)
    results_data = []

    # --- 4. ROI Average THEN Load Scalar & Transform ---
    print("Processing ROIs with saved StandardScalers...")
    
    for roi_name, mask in roi_masks.items():
        try:
            # 1. Load the specific scaler for this ROI's best layer
            scalar_path = os.path.join(args.decode_results_dir, "regressors", roi_name, "betas_scalar.joblib")
            scalar = load(scalar_path)
            
            # 2. ROI Average (Raw)
            roi_raw_series = np.mean(full_stack_raw[:, mask], axis=1) # Shape (Total_Events,)
            
            # 3. Apply the Saved Scaler
            # Note: StandardScaler expects (n_samples, n_features). 
            # Since our series is 1D (averaging over vertices), we reshape to (-1, 1)
            roi_z_series = scalar.transform(roi_raw_series.reshape(-1, 1)).flatten()
            
            # 4. Slice back into metadata chunks
            current_idx = 0
            for meta in metadata_accumulator:
                n_ev = meta['n_events']
                val = roi_z_series[current_idx : current_idx + n_ev]
                current_idx += n_ev
                
                results_data.append({
                    "subject": meta['subject'], "task": meta['task'], "acq": meta['acq'],
                    "run": meta['run'], "tc": meta['tc'], "roi": roi_name,
                    "betas": val.tolist(), "n_repeats": meta['n_repeats']
                })
        except Exception as e:
            print(f"Skipping ROI {roi_name} due to error: {e}")

    # --- 5. Save ---
    if results_data:
        df = pd.DataFrame(results_data)
        os.makedirs(args.save_path, exist_ok=True)
        out_csv = os.path.join(args.save_path, f"{args.subj}_roi_pretrained_zscored_betas.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved results using pretrained scalers to: {out_csv}")

if __name__ == "__main__":
    main()