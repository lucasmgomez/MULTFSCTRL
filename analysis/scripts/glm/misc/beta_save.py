import os
import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib
import argparse
import ast

# ... [Helpers parse_taskdir_name, tc_format, get_trial_betas, 
#      load_atlas_data, create_roi_mask remain the same] ...

# (Keeping your helper functions as they were)
TASKDIR_RE = re.compile(r"^task-(?P<task>[^_]+)_acq-(?P<acq>.+)_run-(?P<run>\d+)$")

def parse_taskdir_name(taskdir_basename: str):
    m = TASKDIR_RE.match(taskdir_basename)
    if not m: return None
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

def load_atlas_data(dlabel_path):
    dl = nib.load(dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    label_dict = dl.header.get_axis(0).label[0] 
    return label_dict, data

def create_roi_mask(dlabel_info, roi_name, lateralize='LR'):
    label_dict, data = dlabel_info
    if lateralize == 'LR': target_labels = [f'L_{roi_name}_ROI', f'R_{roi_name}_ROI']
    elif lateralize == 'L': target_labels = [f'L_{roi_name}_ROI']
    elif lateralize == 'R': target_labels = [f'R_{roi_name}_ROI']
    else: target_labels = [roi_name]
    matched_keys = [k for k, (name, _) in label_dict.items() if name in target_labels]
    return np.isin(data, matched_keys)

# ---------------------------------------------------------
# 3. Main Processing Logic (Modified)
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract Individual Trial/Event ROI Betas")
    # ... (Arguments remain same as your original script) ...
    parser.add_argument("--behav_dir", type=str, default="/mnt/tempdata/lucas/fmri/recordings/TR/behav")
    parser.add_argument("--betas_dir", type=str, default="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lsa_wfdelay/64kDense")
    parser.add_argument("--dlabel_path", type=str, default="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii")
    parser.add_argument("--save_path", type=str, default="./roi_results")
    parser.add_argument("--subj", type=str, default="sub-01")
    parser.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02", "ses-03", "ses-04"])
    parser.add_argument("--events_type", type=str, default='wfdelay')
    parser.add_argument("--lateralize", type=str, default='LR', choices=['LR', 'L', 'R'])
    default_rois = ['10pp','10v','47s','46','9-46d'] # Shortened for example
    parser.add_argument("--rois", type=str, default=str(default_rois))

    args = parser.parse_args()

    try:
        roi_list = ast.literal_eval(args.rois)
    except Exception:
        roi_list = default_rois

    atlas_info = load_atlas_data(args.dlabel_path)
    roi_masks = {roi: create_roi_mask(atlas_info, roi, args.lateralize) for roi in roi_list}
    roi_masks = {k: v for k, v in roi_masks.items() if v.sum() > 0}

    betas_storage = {}
    subj_betas_dir = os.path.join(args.betas_dir, args.subj)
    subj_behav_dir = os.path.join(args.behav_dir, args.subj)

    for session in args.sessions:
        print(f"\nProcessing {args.subj} {session}...")
        betas_ses_dir = os.path.join(subj_betas_dir, session)
        behav_ses_dir = os.path.join(subj_behav_dir, session)
        if not os.path.exists(betas_ses_dir): continue

        files = [d for d in os.listdir(betas_ses_dir) if d.startswith("task-")]
        for f in files:
            print(f"  Checking {f}...")
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
                    if tc not in betas_storage[f]: betas_storage[f][tc] = []
                    betas_storage[f][tc].append(trial_betas)
                    print(f"    Loaded betas for trial {trial} (tc={tc}) with shape {trial_betas.shape}")
                except Exception: continue

    # --- 3. Extract ROI values WITHOUT Averaging ---
    print("\nExtracting all trial/event values...")
    results_data = []

    for f_name, tc_dict in betas_storage.items():
        print(f"Processing {f_name} with {len(tc_dict)} unique trial types...")
        parsed = parse_taskdir_name(f_name)
        if not parsed: continue
        task, acq, run = parsed

        for tc, trial_list in tc_dict.items():
            print(f"  Trial type {tc} has {len(trial_list)} repeats")
            # trial_list is a list of arrays, each (n_events, n_vertices)
            for repeat_idx, trial_data in enumerate(trial_list):
                n_events = trial_data.shape[0]
                
                for event_idx in range(1, min(n_events, 10), 2):
                    event_map = trial_data[event_idx, :] # (n_vertices,)

                    for roi_name, mask in roi_masks.items():
                        val = np.mean(event_map[mask])
                        
                        results_data.append({
                            "subject": args.subj,
                            "task": task,
                            "acq": acq,
                            "run": run,
                            "tc": tc,
                            "repeat_index": repeat_idx,
                            "event_index": event_idx,
                            "roi": roi_name,
                            "beta": val
                        })

    if results_data:
        df = pd.DataFrame(results_data)
        os.makedirs(args.save_path, exist_ok=True)
        out_csv = os.path.join(args.save_path, f"{args.subj}_roi_all_trials.csv")
        df.to_csv(out_csv, index=False)
        print(f"Done! Saved {len(df)} rows to: {out_csv}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()