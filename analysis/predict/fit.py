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

def select_data(betas, acts, phase2predict='encoding', save_per_run=False, cache_dir=None, return_runwise=False):
    stacked_s_betas = []
    stacked_s_acts = []
    runwise_data = {}
    
    if save_per_run and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    for run_name in betas.keys():
        run_betas_list = []
        run_acts_list = []
        
        for tc in sorted(betas[run_name].keys()):
            selected_betas = betas[run_name][tc]
            if tc not in acts[run_name]: 
                continue 
            selected_acts = acts[run_name][tc]

            if '1back' in run_name:
                selected_betas = selected_betas[:10, :]  
            n_betas = selected_betas.shape[0]

            n_tokens = selected_acts.shape[1]
            start_idx = max(0, n_tokens - n_betas)
            selected_acts = selected_acts[:, start_idx:] 

            encoding_idxs = [i for i in range(0, selected_acts.shape[1], 2)]
            delay_idxs = [i for i in range(1, selected_acts.shape[1], 2)]
            
            if len(encoding_idxs) == 0 or len(delay_idxs) == 0: continue

            if phase2predict == 'encoding':
                selected_betas = selected_betas[::2, :] 
            elif phase2predict == 'delay':
                selected_betas = selected_betas[1::2, :]  
                encoding_idxs = encoding_idxs[:len(delay_idxs)]

            enc_acts = selected_acts[:, encoding_idxs, :]
            delay_acts = selected_acts[:, delay_idxs, :]
            selected_acts = np.concatenate((enc_acts, delay_acts), axis=-1)
            
            run_betas_list.append(selected_betas)
            run_acts_list.append(selected_acts)

        if not run_betas_list:
            continue
            
        run_betas_stack = np.concatenate(run_betas_list, axis=0)
        run_acts_stack = np.concatenate(run_acts_list, axis=1)
        
        if save_per_run and cache_dir:
            b_path = os.path.join(cache_dir, f"{run_name}_selected_betas.npy")
            a_path = os.path.join(cache_dir, f"{run_name}_selected_acts.npy")
            np.save(b_path, run_betas_stack)
            np.save(a_path, run_acts_stack)
            print(f"Saved per-run cache: {run_name}")

        runwise_data[run_name] = {
            'betas': run_betas_stack,
            'acts': run_acts_stack
        }

        stacked_s_betas.append(run_betas_stack)
        stacked_s_acts.append(run_acts_stack)

    if not stacked_s_betas:
        raise ValueError("No data found after selection process.")

    if return_runwise:
        return runwise_data

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
        
    return mask

def predict(betas, acts, model, avg_vertices, standardize_acts, standardize_betas, **model_kwargs):
    # Pass through the raw tuple since it changes length based on the model used
    return model(acts, betas, avg_vertices=avg_vertices, standardize_acts=standardize_acts, standardize_betas=standardize_betas, **model_kwargs)

