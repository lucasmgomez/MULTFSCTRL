import os
import pandas as pd
import re
import numpy as np
import glob
import nibabel as nib
import torch
import json
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
        beta_file = glob.glob(beta_file_pattern)[0]
        b = nib.load(beta_file).get_fdata().squeeze().astype(np.float32)
        betas.append(b)

    return np.stack(betas, axis=0)

def get_activations(acts_task_dir, tc):
    acts_dict = torch.load(acts_task_dir)
    all_acts = acts_dict['layer_activations']
    all_tcs = acts_dict['tcs']

    acts = all_acts[:, all_tcs.index(tc)]

    return acts

def tc_format(task, tc):
    task_tc_len_map = {'ctxdm': 6, 'interdms': 8, '1back': 12} 

    correct_len = task_tc_len_map[task]
    if len(str(tc)) < correct_len:
        tc = tc.zfill(correct_len)   

    if task == '1back': # truncates to 5 stimuli (10 loc x obj pairs)
        tc = tc[:10]

    return tc         

def build_data(behav_dir, betas_dir, acts_dir, subj, sessions):
    betas = {}
    acts = {}

    for session in sessions:

        behav_ses_dir = os.path.join(behav_dir, subj, session)
        betas_ses_dir = os.path.join(betas_dir, subj, session)

        files = [d for d in os.listdir(betas_ses_dir) if d.startswith("task-")]

        for f in files:
            betas[f] = {}
            acts[f] = {}

            task, acq, run = parse_taskdir_name(f)

            task_tag = f"task-{task}"

            hit = glob.glob(os.path.join(behav_ses_dir, f"*{task_tag}_{acq}*block_{int(run)-1}*scored*.tsv"))

            if not hit:
                raise FileNotFoundError(f"Missing block log for {subj} {session} {task_tag} run-{run}")
        
            block_log_tsv = hit[0]
            block_df = pd.read_csv(block_log_tsv, sep="\t")
            block_df['TrialNumber'] = block_df['TrialNumber'].astype(int)
            block_df['tc'] = block_df['tc'].astype(int)

            events_tsv = os.path.join(behav_ses_dir, f"events/{task_tag}_{acq}_run-{run}_base-events.tsv")
            events = pd.read_csv(events_tsv, sep="\t")
            events = events.sort_values(by=['onset']).reset_index(drop=True)
                                      
            betas_task_dir = os.path.join(betas_ses_dir, f"{task_tag}_acq-{acq}_run-{run}")
            acts_task_dir = os.path.join(acts_dir, f"{task_tag}_{acq}.pth")

            for trial, tc in zip(block_df['TrialNumber'], block_df['tc']):
                tc = tc_format(task, str(tc))
                
                # Collect betas
                trial_betas = get_betas(betas_task_dir, events, trial)
                if tc not in betas[f]:
                    betas[f][tc] = [trial_betas]
                else:
                    betas[f][tc].append(trial_betas)

                # Collect activations
                trial_acts = get_activations(acts_task_dir, tc)
                acts[f][tc] = trial_acts.numpy()

    # Average betas over session repeats
    for f in betas.keys():
        for tc in betas[f].keys():
            betas[f][tc] = np.mean(np.stack(betas[f][tc], axis=0), axis=0)
    
    return betas, acts

def make_roi_mask(dlabel_path: str, roi_names):
    dl = nib.load(dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    label_dict = dl.header.get_axis(0).label[0]  # {id: (name, rgba)}

    roi_set = {r.strip() for r in roi_names if r.strip()}
    if not roi_set:
        raise ValueError("roi_names is empty after parsing.")

    keys = [k for k, (name, _) in label_dict.items() if name in roi_set]
    found = [label_dict[k][0] for k in keys]
    missing = sorted(list(roi_set - set(found)))
    if missing:
        all_names = sorted([name for _, (name, _) in label_dict.items()])
        suggestions = [n for n in all_names if any(m.lower() in n.lower() for m in missing)][:40]
        raise ValueError(
            f"Missing ROI labels: {missing}\n"
            f"Found: {found}\n"
            f"Suggestions: {suggestions}"
        )

    mask = np.isin(data, keys)
    if mask.sum() == 0:
        raise ValueError("ROI mask has 0 vertices (space mismatch or wrong labels).")

    roi_label = "+".join(found)
    print("Matched ROI labels:", found)
    print("Total ROI vertices:", int(mask.sum()))
    return mask, roi_label

def create_beta_mask(dlabel_info, roi, lateralize):
    label_dict, data = dlabel_info

    # Format rois
    if lateralize == 'LR':
        roi = ['L_' + roi + '_ROI'] + ['R_' + roi + '_ROI']
    elif lateralize == 'L':
        roi = ['L_' + roi + '_ROI']
    elif lateralize == 'R':
        roi = ['R_' + roi + '_ROI']

    keys = [k for k, (name, _) in label_dict.items() if name in roi]
    mask = np.isin(data, keys)
    if mask.sum() == 0:
        raise ValueError("ROI mask has 0 vertices (space mismatch or wrong labels).")

    return mask

def select_data(betas, acts, phase2predict = 'encoding'):
    stacked_s_betas = []
    stacked_s_acts = []

    for run in betas.keys():
        for tc in betas[run].keys():
            selected_betas = betas[run][tc]
            selected_acts = acts[run][tc]

            # Beta processing
            if '1back' in run:
                selected_betas = selected_betas[:10, :]  # select first 10 betas for 1back task
            n_betas = selected_betas.shape[0]

            if phase2predict == 'encoding':
                selected_betas = selected_betas[::2, :] 
            elif phase2predict == 'delay':
                selected_betas = selected_betas[1::2, :]  

            n_tokens = selected_acts.shape[1]
            selected_acts = selected_acts[:, n_tokens-n_betas:] # NOTE: select images only
            # ins_acts = np.mean(selected_acts[:, :n_tokens-10], axis=1)   NOTE: not used currently

            # NOTE: first try we only joining encoding and delay activations
            encoding_idxs = [i for i in range(0, selected_acts.shape[1], 2)]
            delay_idxs = [i for i in range(1, selected_acts.shape[1], 2)]
            enc_acts = selected_acts[:, encoding_idxs, :]
            delay_acts = selected_acts[:, delay_idxs, :]
            selected_acts = np.concatenate((enc_acts, delay_acts), axis=-1)

            stacked_s_betas.append(selected_betas)
            stacked_s_acts.append(selected_acts)

    s_betas = np.concatenate(stacked_s_betas, axis=0)
    s_acts = np.concatenate(stacked_s_acts, axis=1)

    return s_betas, s_acts

def predict(betas, acts, model, avg_vertices, standardize_acts, standardize_betas, **model_kwargs):
    
    result, _, _ = model(acts, betas, avg_vertices=avg_vertices, standardize_acts=standardize_acts, standardize_betas=standardize_betas, **model_kwargs)

    return

def main():
    
    behav_dir = "/mnt/tempdata/lucas/fmri/recordings/TR/behav"
    betas_dir = "/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lss/64kDense"
    acts_dir = "/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/activations"
    dlabel_path = "/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii"
    save_path = "/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/test"
    run_name = "test_run"

    subj = "sub-01"
    sessions = ["ses-01", "ses-02", "ses-03", "ses-04"]
    rois = ['SFL', 'i6-8', 's6-8', 'IFJa', 'IFJp', 'IFSp', 'IFSa', '8BM', '8Av', '8Ad', '8BL', '8C', 
            '9m', '9p', '9a', '9-46d', 'a9-46v', 'p9-46v', '46', '44', '45', '47l', '47m', '47s', 
            'a47r', 'p47r', '10r', '10d', '10v', 'a10p', 'p10p', '10pp', '11l', '13l'] # list of rois, if network it should be "network"
    lateralize = 'LR' # Whether to lateralize ROIs
    betas, acts = build_data(behav_dir, betas_dir, acts_dir, subj, sessions)


    s_betas, s_acts = select_data(betas, acts, phase2predict = 'encoding')

    # Load glasser dlabel info
    dl = nib.load(dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    label_dict = dl.header.get_axis(0).label[0] 

    # Results dict
    results = {}

    for roi in rois:
        # Create and apply beta mask
        mask = create_beta_mask((label_dict, data), roi, lateralize)
        m_s_betas = s_betas[:, mask]
        import pdb; pdb.set_trace()

        # Predict
        result, _ , _ = predict(m_s_betas, s_acts, model=pca_ridge_decode, avg_vertices=True, standardize_acts=True, standardize_betas=True, ridge_alpha=0.5, n_pcs=64)
        results[roi] = result

    # Save the decoded activations to a JSON file
    folder_path = f"{save_path}/{run_name}/"
    os.makedirs(os.path.dirname(folder_path), exist_ok=True)
    with open(folder_path + 'results.json', 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()






        
        

