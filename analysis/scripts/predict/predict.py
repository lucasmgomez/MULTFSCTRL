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
    trial_events = base_events[base_events['TrialNumber'] == trial_num]

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

    acts = all_acts[all_tcs.index(tc)]

    return acts


def build_data(behav_dir, betas_dir, acts_dir, subj, sessions):
    betas = {}
    acts = {}

    for session in sessions:
        betas[session] = {}

        behav_ses_dir = os.path.join(behav_dir, subj, session)
        betas_ses_dir = os.path.join(betas_dir, subj, session)

        runs = [d for d in os.listdir(betas_ses_dir) if d.startswith("task-")]

        for run in runs:
            betas[run] = {}
            acts[run] = {}

            task, acq, run = parse_taskdir_name(run)

            task_tag = f"task-{task}"

            hit = glob.glob(os.path.join(behav_ses_dir, f"*{task_tag}_{acq}*block_{int(run)-1}*scored*.tsv"))

            if not hit:
                raise FileNotFoundError(f"Missing block log for {subj} {session} {task_tag} run-{run}")
        
            block_log_tsv = hit[0]
            block_df = pd.read_csv(block_log_tsv, sep="\t")
            block_df['TrialNumber'] = block_df['TrialNumber'].astype(int)

            events_tsv = os.path.join(behav_ses_dir, f"{task_tag}_{acq}_run-{run}_base-events.tsv")
            events = pd.read_csv(events_tsv, sep="\t")
            events = events.sort_values(by=['onset']).reset_index(drop=True)
                                      
            betas_task_dir = os.path.join(betas_ses_dir, f"{task_tag}_acq-{acq}_run-{run}_betas")
            acts_task_dir = os.path.join(acts_dir, subj, session, f"{task_tag}_{acq}.pth")

            for trial, tc in zip(block_df['TrialNumber'], block_df['tc']):
                
                # Collect betas
                trial_betas = get_betas(betas_task_dir, events, trial)
                if tc not in betas[run]:
                    betas[run][tc] = []
                else:
                    betas[run][tc].append(trial_betas)

                # Collect activations
                trial_acts = get_activations(acts_task_dir, tc)
                acts[run][tc] = trial_acts
            
    return betas, acts

def main():
    
    behav_dir = "/mnt/tempdata/lucas/fmri/recordings/TR/behav"
    betas_dir = "/mnt/tempdata/lucas/fmri/analyses/TR/glm/betas"
    acts_dir = "/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/activations"

    subj = "sub-01"
    sessions = ["ses-01", "ses-02", "ses-03", "ses-04"]

    betas, acts = build_data(behav_dir, betas_dir, acts_dir, subj, sessions)





        
        

