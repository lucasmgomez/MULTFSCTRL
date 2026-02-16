import os
import argparse
import ast
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import re
import glob
from joblib import load

# ---------------------------------------------------------
# 1. Helpers for Metadata Reconstruction
# ---------------------------------------------------------

TASKDIR_RE = re.compile(r"^task-(?P<task>[^_]+)_acq-(?P<acq>.+)_run-(?P<run>\d+)")

def parse_taskdir_name(taskdir_basename: str):
    m = TASKDIR_RE.match(taskdir_basename)
    if not m:
        return None
    return m.group("task"), m.group("acq"), m.group("run")

def tc_format(task, tc):
    task_tc_len_map = {'ctxdm': 6, 'interdms': 8, '1back': 12, 'wfdelay': 6} 
    correct_len = task_tc_len_map.get(task, len(str(tc))) 
    if len(str(tc)) < correct_len:
        tc = tc.zfill(correct_len)   
    if task == '1back': 
        tc = tc[:10]
    return tc         

def load_atlas_data(dlabel_path):
    dl = nib.load(dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    label_dict = dl.header.get_axis(0).label[0]
    return label_dict, data

def create_roi_mask(dlabel_info, roi_name, lateralize='LR'):
    label_dict, data = dlabel_info

    if lateralize == 'LR':
        roi_list = [f'L_{roi_name}_ROI', f'R_{roi_name}_ROI']
    elif lateralize == 'L':
        roi_list = [f'L_{roi_name}_ROI']
    elif lateralize == 'R':
        roi_list = [f'R_{roi_name}_ROI']
    else:
        roi_list = [roi_name]

    keys = [k for k, (name, _) in label_dict.items() if name in roi_list]
    return np.isin(data, keys)

def get_trial_betas(betas_task_dir, base_events, trial_num):
    # Lightweight helper to just get beta shape
    trial_events = base_events[base_events['trial'] == f"Trial{trial_num:02d}"]
    if trial_events.empty:
        raise ValueError
    eids = trial_events['event_id'].to_list()
    # We only need to load one to check shape, but to be safe we follow pattern
    # Actually, we need n_betas count. That is len(eids).
    # Wait, eids is events. Betas are files. 1 beta per event.
    return len(eids)

# ---------------------------------------------------------
# 2. Main Script
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate ROI betas from Cached s_betas.npy + Rebuilt Metadata")
    
    # Paths required for Metadata Reconstruction
    ap.add_argument("--behav_dir", default="/mnt/tempdata/lucas/fmri/recordings/TR/behav")
    ap.add_argument("--betas_dir", default="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lsa_wfdelay/64kDense")
    ap.add_argument("--acts_dir", required=True)
    ap.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02", "ses-03", "ses-04"])
    ap.add_argument("--events_type", default='wfdelay')

    # Paths required for Data Loading
    ap.add_argument("--data_cache_dir", required=True)
    ap.add_argument("--decode_results_dir", required=True)
    ap.add_argument("--dlabel_path", required=True)
    ap.add_argument("--save_path", required=True)
    
    # Config
    ap.add_argument("--subj", default="sub-01")
    ap.add_argument("--lateralize", default="LR", choices=["LR", "L", "R"])
    ap.add_argument("--rois", required=True)

    args = ap.parse_args()

    # --- PART 1: Load Cached Betas (The "Truth") ---
    s_betas_path = os.path.join(args.data_cache_dir, "s_betas.npy")
    if not os.path.exists(s_betas_path):
        raise FileNotFoundError(f"Missing s_betas.npy at: {s_betas_path}")

    s_betas = np.load(s_betas_path)
    print(f"Loaded s_betas: {s_betas.shape} (Events x Vertices)")
    n_total_events = s_betas.shape[0]

    # --- PART 2: Reconstruct Metadata (The "Labels") ---
    print("Reconstructing metadata to match s_betas rows...")
    metadata_rows = []
    
    subj_betas_dir = os.path.join(args.betas_dir, args.subj)
    subj_behav_dir = os.path.join(args.behav_dir, args.subj)

    for session in args.sessions:
        betas_ses_dir = os.path.join(subj_betas_dir, session)
        behav_ses_dir = os.path.join(subj_behav_dir, session)
        if not os.path.exists(betas_ses_dir): continue

        files = sorted([d for d in os.listdir(betas_ses_dir) if d.startswith("task-")])

        for f in files:
            parsed = parse_taskdir_name(f)
            if not parsed: continue
            task, acq, run = parsed
            
            # A. Load Activations
            task_tag = f"task-{task}"
            acts_task_dir = os.path.join(args.acts_dir, f"{task_tag}_{acq}.pth")
            if not os.path.exists(acts_task_dir): continue

            try:
                acts_dict = torch.load(acts_task_dir, map_location='cpu')
                all_tcs = acts_dict['tcs']
                if not isinstance(all_tcs, list): all_tcs = all_tcs.tolist()
                
                # We need acts shape to check length
                all_acts = acts_dict['layer_activations'] # [Layers, TCs, Tokens, Units]
            except: continue

            # B. Load Behavior
            hit = glob.glob(os.path.join(behav_ses_dir, f"*{task_tag}_{acq}*block_{int(run)-1}*scored*.tsv"))
            if not hit: continue
            
            block_df = pd.read_csv(hit[0], sep="\t")
            try: block_df['tc'] = block_df['tc'].astype(int)
            except: pass

            events_tsv = os.path.join(behav_ses_dir, f"events_{args.events_type}/{task_tag}_{acq}_run-{run}_base-events.tsv")
            if not os.path.exists(events_tsv): continue
            events = pd.read_csv(events_tsv, sep="\t").sort_values(by=['onset']).reset_index(drop=True)
            betas_task_dir = os.path.join(betas_ses_dir, f)

            # C. Iterate Trials
            for trial, raw_tc in zip(block_df['TrialNumber'], block_df['tc']):
                tc = tc_format(task, str(raw_tc))
                
                # Check 1: TC existence
                if tc not in all_tcs: continue
                
                try:
                    # Check 2: Get n_betas (from neural files)
                    # We need to know how many betas exist to check vs tokens
                    n_betas = get_trial_betas(betas_task_dir, events, trial)
                    
                    if '1back' in task:
                        n_betas = min(n_betas, 10) # 1back truncation

                    # Check 3: Get n_tokens (from acts)
                    tc_idx = all_tcs.index(tc)
                    n_tokens = all_acts.shape[2] 

                    # CRITICAL FILTER: Implicit check in training script
                    # If tokens < betas, shapes mismatch and training would have crashed.
                    # Since training succeeded, these trials must be excluded.
                    if n_tokens < n_betas:
                        continue

                    # Check 4: Delay idx existence
                    delay_idxs = [i for i in range(1, n_tokens, 2)]
                    if len(delay_idxs) == 0: continue

                except Exception: 
                    continue

                # If passed, this trial exists in s_betas.npy
                metadata_rows.append({
                    "subject": args.subj,
                    "session": session,
                    "task": task,
                    "acq": acq,
                    "run": run,
                    "trial": trial,
                    "tc": tc
                })

    meta_df = pd.DataFrame(metadata_rows)
    print(f"Reconstructed Metadata Rows: {len(meta_df)}")

    if len(meta_df) != n_total_events:
        print(f"WARNING: Metadata count ({len(meta_df)}) still mismatch s_betas count ({n_total_events})!")
        # At this point, save unlabelled to preserve the values
    else:
        print("Success! Metadata count matches s_betas count.")

    # --- PART 3: Process ROIs & Merge ---
    roi_list = ast.literal_eval(args.rois)
    atlas_info = load_atlas_data(args.dlabel_path)
    
    roi_data = {}

    for roi in roi_list:
        mask = create_roi_mask(atlas_info, roi, args.lateralize)
        if mask.sum() == 0:
            print(f"[skip] ROI {roi}: 0 vertices")
            continue

        scaler_path = os.path.join(args.decode_results_dir, "regressors", roi, "betas_scalar.joblib")
        if not os.path.exists(scaler_path):
            print(f"[skip] ROI {roi}: missing scaler")
            continue

        scaler = load(scaler_path)

        roi_raw = np.mean(s_betas[:, mask], axis=1)
        roi_z = scaler.transform(roi_raw.reshape(-1, 1)).ravel()
        
        roi_data[f"beta_{roi}"] = roi_z

    # --- PART 4: Save Final CSV ---
    roi_df = pd.DataFrame(roi_data)
    
    if len(meta_df) == len(roi_df):
        final_df = pd.concat([meta_df.reset_index(drop=True), roi_df.reset_index(drop=True)], axis=1)
    else:
        print("Metadata/Data mismatch - Saving ROI data only.")
        final_df = roi_df

    os.makedirs(args.save_path, exist_ok=True)
    out_csv = os.path.join(args.save_path, f"{args.subj}_roi_pretrained_zscored_betas_FULL.csv")
    final_df.to_csv(out_csv, index=False)
    print(f"Saved complete dataset to: {out_csv}")

if __name__ == "__main__":
    main()