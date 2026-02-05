import os
import shutil
import pandas as pd
import re
import numpy as np
import glob
import nibabel as nib
import torch
import json
import argparse
import ast
from joblib import dump
from models.regression import pca_ridge_decode

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

def get_activations(acts_task_dir, tc):
    acts_dict = torch.load(acts_task_dir)
    all_acts = acts_dict['layer_activations']
    all_tcs = acts_dict['tcs']

    acts = all_acts[:, all_tcs.index(tc)]
    return acts.numpy()

def tc_format(task, tc):
    task_tc_len_map = {'ctxdm': 6, 'interdms': 8, '1back': 12} 

    correct_len = task_tc_len_map.get(task, len(str(tc))) 
    if len(str(tc)) < correct_len:
        tc = tc.zfill(correct_len)   

    if task == '1back': # truncates to 5 stimuli (10 loc x obj pairs)
        tc = tc[:10]

    return tc         

def build_data(behav_dir, betas_dir, acts_dir, subj, sessions, events_type='base'):
    betas = {}
    acts = {}

    for session in sessions:
        behav_ses_dir = os.path.join(behav_dir, subj, session)
        betas_ses_dir = os.path.join(betas_dir, subj, session)

        if not os.path.exists(betas_ses_dir):
            print(f"Warning: Directory not found: {betas_ses_dir}")
            continue

        files = [d for d in os.listdir(betas_ses_dir) if d.startswith("task-")]

        for f in files:
            betas[f] = {}
            acts[f] = {}

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
            acts_task_dir = os.path.join(acts_dir, f"{task_tag}_{acq}.pth")

            if not os.path.exists(acts_task_dir):
                print(f"Skipping missing activations: {acts_task_dir}")
                continue

            for trial, tc in zip(block_df['TrialNumber'], block_df['tc']):
                tc = tc_format(task, str(tc))
                
                # Collect betas
                try:
                    trial_betas = get_betas(betas_task_dir, events, trial)
                    if tc not in betas[f]:
                        betas[f][tc] = [trial_betas]
                    else:
                        betas[f][tc].append(trial_betas)

                    # Collect activations
                    trial_acts = get_activations(acts_task_dir, tc)
                    acts[f][tc] = trial_acts
                except Exception as e:
                    print(f"Error processing {f} trial {trial}: {e}")
                    continue

    # Average betas over session repeats
    for f in betas.keys():
        for tc in list(betas[f].keys()): 
            if len(betas[f][tc]) > 0:
                betas[f][tc] = np.mean(np.stack(betas[f][tc], axis=0), axis=0)
    
    return betas, acts

def select_data(betas, acts, phase2predict='encoding'):
    stacked_s_betas = []
    stacked_s_acts = []

    for run in betas.keys():
        for tc in betas[run].keys():
            selected_betas = betas[run][tc]
            if tc not in acts[run]: 
                continue 
            selected_acts = acts[run][tc]

            # Beta processing
            if '1back' in run:
                selected_betas = selected_betas[:10, :]  
            n_betas = selected_betas.shape[0]

            n_tokens = selected_acts.shape[1]
            start_idx = max(0, n_tokens - n_betas)
            selected_acts = selected_acts[:, start_idx:] 

            encoding_idxs = [i for i in range(0, selected_acts.shape[1], 2)]
            delay_idxs = [i for i in range(1, selected_acts.shape[1], 2)]
            
            if len(encoding_idxs) == 0 or len(delay_idxs) == 0: continue

            if  phase2predict == 'encoding':
                selected_betas = selected_betas[::2, :] 
            elif phase2predict == 'delay':
                selected_betas = selected_betas[1::2, :]  
                encoding_idxs = encoding_idxs[:len(delay_idxs)]

            enc_acts = selected_acts[:, encoding_idxs, :]
            delay_acts = selected_acts[:, delay_idxs, :]
            selected_acts = np.concatenate((enc_acts, delay_acts), axis=-1)

            stacked_s_betas.append(selected_betas)
            stacked_s_acts.append(selected_acts)

    if not stacked_s_betas:
        raise ValueError("No data found after selection process.")

    s_betas = np.concatenate(stacked_s_betas, axis=0)
    s_acts = np.concatenate(stacked_s_acts, axis=1)

    return s_betas, s_acts

def make_roi_mask(dlabel_path: str, roi_names):
    dl = nib.load(dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    label_dict = dl.header.get_axis(0).label[0] 

    roi_set = {r.strip() for r in roi_names if r.strip()}
    keys = [k for k, (name, _) in label_dict.items() if name in roi_set]
    found = [label_dict[k][0] for k in keys]
    
    mask = np.isin(data, keys)
    roi_label = "+".join(found)
    return mask, roi_label

def create_beta_mask(dlabel_info, roi, lateralize):
    label_dict, data = dlabel_info

    # Format rois
    if lateralize == 'LR':
        roi_list = ['L_' + roi + '_ROI', 'R_' + roi + '_ROI']
    elif lateralize == 'L':
        roi_list = ['L_' + roi + '_ROI']
    elif lateralize == 'R':
        roi_list = ['R_' + roi + '_ROI']
    else:
        roi_list = [roi]

    keys = [k for k, (name, _) in label_dict.items() if name in roi_list]
    mask = np.isin(data, keys)
    
    if mask.sum() == 0:
        # print(f"Warning: ROI mask for {roi} has 0 vertices.")
        pass
        
    return mask


def predict(betas, acts, model, avg_vertices, standardize_acts, standardize_betas, **model_kwargs):
    result, _, regressor = model(acts, betas, avg_vertices=avg_vertices, standardize_acts=standardize_acts, standardize_betas=standardize_betas, **model_kwargs)
    return result, _, regressor

def main():
    parser = argparse.ArgumentParser(description="fMRI PCA Ridge Decoding")

    # Path arguments
    parser.add_argument("--behav_dir", type=str, default="/mnt/tempdata/lucas/fmri/recordings/TR/behav",
                        help="Path to behavior directory")
    parser.add_argument("--betas_dir", type=str, default="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lss/64kDense",
                        help="Path to betas directory")
    parser.add_argument("--acts_dir", type=str, default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/activations",
                        help="Path to activations directory")
    parser.add_argument("--dlabel_path", type=str, default="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii",
                        help="Path to dlabel nifti file")
    parser.add_argument("--save_path", type=str, default="/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/test",
                        help="Root path to save results")
    
    # Run configuration
    parser.add_argument("--run_name", type=str, default="test_run", help="Name of the current run/experiment")
    parser.add_argument("--subj", type=str, default="sub-01", help="Subject ID")
    parser.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02", "ses-03", "ses-04"],
                        help="List of sessions (space separated)")
    parser.add_argument("--events_type", type=str, default='wfdelay', choices=['wfdelay', 'wofdelay'], 
                        help="Type of events to use for building data")
    parser.add_argument("--lateralize", type=str, choices=['LR', 'L', 'R'], default='LR',
                        help="Lateralization of ROIs")
    parser.add_argument("--phase2predict", type=str, default='delay', choices=['encoding', 'delay'],
                        help="Phase to predict")
    parser.add_argument("--standardize_betas", action='store_true', 
                        help="If set, standardizes betas before decoding.")
    parser.add_argument("--standardize_acts", action='store_true', 
                        help="If set, standardizes activations before decoding.")   
    
    # Cache / Preprocessing Arguments
    parser.add_argument("--save_data", action='store_true', 
                        help="If set, saves the concatenated s_betas and s_acts to disk after processing.")
    parser.add_argument("--load_data", action='store_true',
                        help="If set, loads s_betas and s_acts from disk instead of reprocessing raw files.")
    parser.add_argument("--data_cache_dir", type=str, default=None,
                        help="Directory to save/load cached data. Defaults to save_path/run_name if not specified.")

    # ROI list
    default_rois_list = [
        'SFL', 'i6-8', 's6-8', 'IFJa', 'IFJp', 'IFSp', 'IFSa', '8BM', '8Av', '8Ad', '8BL', '8C', 
        '9m', '9p', '9a', '9-46d', 'a9-46v', 'p9-46v', '46', '44', '45', '47l', '47m', '47s', 
        'a47r', 'p47r', '10r', '10d', '10v', 'a10p', 'p10p', '10pp', '11l', '13l'
    ]
    parser.add_argument("--rois", type=str, default=str(default_rois_list), 
                        help="String representation of a python list of ROIs.")
    
    args = parser.parse_args()

    # Define results folder
    folder_path = os.path.join(args.save_path, args.run_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Define Cache Directory
    # If explicit path provided, use it. Otherwise use the results folder.
    cache_dir = args.data_cache_dir if args.data_cache_dir else folder_path

    # --- BLOCK 1: Load or Build Data ---
    if args.load_data:
        print(f"Loading preprocessed data from: {cache_dir}")
        try:
            s_betas = np.load(os.path.join(cache_dir, 's_betas.npy'))
            s_acts = np.load(os.path.join(cache_dir, 's_acts.npy'))
            print(f"Data loaded successfully. Shapes: Betas {s_betas.shape}, Acts {s_acts.shape}")
        except FileNotFoundError as e:
            print(f"Error: Could not find cached data files in {cache_dir}.")
            print(f"Details: {e}")
            return
    else:
        print(f"Processing Subject: {args.subj}")
        print(f"Sessions: {args.sessions}")
        
        # Build and Select
        betas, acts = build_data(args.behav_dir, args.betas_dir, args.acts_dir, args.subj, args.sessions, events_type=args.events_type)
        s_betas, s_acts = select_data(betas, acts, phase2predict=args.phase2predict)
        
        # Save if requested
        if args.save_data:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Saving processed data to: {cache_dir}")
            np.save(os.path.join(cache_dir, 's_betas.npy'), s_betas)
            np.save(os.path.join(cache_dir, 's_acts.npy'), s_acts)

    # --- BLOCK 2: Run ROI Decoding ---
    
    # Parse ROI string
    try:
        rois = ast.literal_eval(args.rois)
    except Exception as e:
        print(f"Error parsing ROI string: {e}")
        return

    # Load glasser dlabel info
    dl = nib.load(args.dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    label_dict = dl.header.get_axis(0).label[0] 

    results = {}
    regressors = {}
    print(f"Starting decoding for {len(rois)} ROIs...")
    
    for roi in rois:
        mask = create_beta_mask((label_dict, data), roi, args.lateralize)
        if mask.sum() == 0:
            continue    
        
        curr_betas = s_betas[:, mask]
        
        try:
            result, _ , regressor = predict(
                curr_betas, s_acts, 
                model=pca_ridge_decode, 
                avg_vertices=True, 
                standardize_acts=args.standardize_acts,
                standardize_betas=args.standardize_betas,
                ridge_alpha=0.5, 
                n_pcs=64
            )
            results[roi] = result
            regressors[roi] = regressor
            print(f"Finished {roi}")
        except Exception as e:
            print(f"Failed decoding {roi}: {e}")

    # Save Results
    save_file = os.path.join(folder_path, 'results.json')
    with open(save_file, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {save_file}")

    # Save Regressors
    regressors_path = os.path.join(folder_path, 'regressors')
    for k, regs in regressors.items():
        regressor_path = os.path.join(regressors_path, k)
        if not os.path.exists(regressor_path):
            os.makedirs(regressor_path)
        else:
            shutil.rmtree(regressor_path)
            os.makedirs(regressor_path)
        for layer_idx, reg in enumerate(regs):
            dump(reg, os.path.join(regressor_path, f'layer_{layer_idx}.joblib'))

if __name__ == "__main__":
    main()