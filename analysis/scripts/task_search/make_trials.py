from base_tasks import DMSLoc, DMSCategory, DMSObject, DMSA_LC, DMSA_LO, DMSO_LC, DMSO_LO
from base_tasks import ctxDM_COC, ctxDM_OLC, ctxDM_LOL
from base_tasks import make_oneback_and_or, make_twoback, make_interdms

import sys
sys.path.append('/home/lucas/projects/iWISDM')

from iwisdm import make
from ast import literal_eval
import os
import shutil
import json

from iwisdm import read_write
stim_dir = '/mnt/tempdata/lucas/multi_task_data/stim/only_fmri_zoom' # 

# Create environment
env = make(
    env_id='ShapeNet',
    dataset_fp=stim_dir,
)

""" Helper functions """
def map_loc_LoR(loc):
    x,_ = literal_eval(loc)

    if x < 0.5:
        return 0
    else:
        return 1

def collect_tcs(trials_dir):
    tcs = []
    for trial_folder in os.listdir(trials_dir):
        with open(trials_dir + '/' + trial_folder + '/frames/task_info.json', 'r') as f:
            trial_info = json.load(f)

        tc = "_".join([str(map_loc_LoR(trial_info['objects'][i]['obj']['location'])) +
                         str(trial_info['objects'][i]['obj']['object']) for i in range(len(trial_info['objects']))])
        tcs.append(tc)
    return tcs

def check_tcs(trial_info, tcs):

    new_tc = "_".join([str(map_loc_LoR(trial_info['objects'][i]['obj']['location'])) +
                         str(trial_info['objects'][i]['obj']['object']) for i in range(len(trial_info['objects']))])
    
    return (new_tc in tcs, new_tc)

def make_trials(task, task_info_func, n_trials, folder_fp, remake=False, **kwargs):

    if os.path.exists(folder_fp):
        if remake:
            shutil.rmtree(folder_fp)
            os.makedirs(folder_fp)
            tcs = []
        else:
            tcs = collect_tcs(folder_fp)
    else:
        os.makedirs(folder_fp)
        tcs = []

    nts = 0 + len(tcs)
    while nts < n_trials:
        task_info = task_info_func(task, **kwargs)

        imgs, _, info_dict = task_info.generate_trial(canvas_size=env.env_spec.canvas_size,
                                                        fixation_cue=True,
                                                        cue_on_action=True,
                                                        stim_data=env.stim_data.splits['train']['data'],
                                                        add_distractor_frame=0,
                                                        add_distractor_time=0,
                                                        return_objset = False
                                                    )

        tcisin, tc = check_tcs(info_dict, tcs)

        if not tcisin:
            tcs.append(tc)

            outpath = folder_fp + '/trial%d' % len(os.listdir(folder_fp))
            read_write.write_trial(imgs, info_dict, outpath)

            with open(outpath + '/frames/task_info.json', 'w') as f:
                json.dump(info_dict, f)

            print('Trials left: %d' % (len(tcs) - len(os.listdir(folder_fp))))
            nts += 1

        else:
            print('Duplicate TC found: %s' % tc)
            print('TCS so far: %s' % tcs)
            continue

    return len(tcs)

if __name__ == '__main__':

    base_dir = "/mnt/store_tiny/lucas/iwisdm_trials/control_trials"

    """
      Create all task infos
    """
    
    # DMS And Or
    def dms_a_o_info(dms_a_o_task,  **kwargs):
        return env.init_compositional_tasks([dms_a_o_task])[0]
    dmsa_lc = DMSA_LC(whens=['last2', 'last0'], reverse=True)
    dmsa_lo = DMSA_LO(whens=['last2', 'last0'], reverse=True)
    dmsa_cl = DMSA_LC(whens=['last2', 'last0'], reverse=True)
    dmsa_ol = DMSA_LO(whens=['last2', 'last0'], reverse=True)
    dmso_lc = DMSO_LC(whens=['last2', 'last0'], reverse=True)
    dmso_lo = DMSO_LO(whens=['last2', 'last0'], reverse=True)
    dmso_cl = DMSO_LC(whens=['last2', 'last0'], reverse=True)
    dmso_ol = DMSO_LO(whens=['last2', 'last0'], reverse=True)

    dms_ntrials = (2*4)**2

    
    # Make New CTXDM
    def ctxdm_info(ctxdm_task, **kwargs):
        return env.init_compositional_tasks([ctxdm_task])[0]
    ctxdm_coc = ctxDM_COC(whens=['last4', 'last2', 'last0'], reverse=True)
    ctxdm_olc = ctxDM_OLC(whens=['last4', 'last2', 'last0'], reverse=True)
    ctxdm_lol = ctxDM_LOL(whens=['last4', 'last2', 'last0'], reverse=True)
    ctxdm_ntrials = (2*4)**3

    # Make New InterDMS
    def interdms_info(interdms_task, **kwargs):
        return make_interdms(env=env, DMS=interdms_task, task_type=kwargs['task_type'])
    interdms_ntrials = (2*4)**5 # NOTE: This is not the full space, but any more and it would take forver to run through the model

    # Make New 1-back And Or
    def oneback_andor_info(oneback_task, **kwargs):
        return make_oneback_and_or(env=env, DMSA=oneback_task, k=kwargs['k'])
    oneback_ntrials = (2*4)**5

    # Make New 2-back
    def twoback_info(twoback_task, **kwargs):
        return make_twoback(env=env, DMS=twoback_task, k=kwargs['k'])
    twoback_ntrials = (2*4)**5

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Task to make trials for')
    
    args = parser.parse_args()

    if args.task == 'DMSAndOr':
        make_trials(dmsa_lc, dms_a_o_info, dms_ntrials, base_dir + '/DMSA_LC')
        make_trials(dmsa_lo, dms_a_o_info, dms_ntrials, base_dir + '/DMSA_LO')
        make_trials(dmsa_cl, dms_a_o_info, dms_ntrials, base_dir + '/DMSA_CL')
        make_trials(dmsa_ol, dms_a_o_info, dms_ntrials, base_dir + '/DMSA_OL')
        make_trials(dmso_lc, dms_a_o_info, dms_ntrials, base_dir + '/DMSO_LC')
        make_trials(dmso_lo, dms_a_o_info, dms_ntrials, base_dir + '/DMSO_LO')
        make_trials(dmso_cl, dms_a_o_info, dms_ntrials, base_dir + '/DMSO_CL')
        make_trials(dmso_ol, dms_a_o_info, dms_ntrials, base_dir + '/DMSO_OL')
    elif args.task == 'CTXDM':
        make_trials(ctxdm_coc, ctxdm_info, ctxdm_ntrials, base_dir + '/ctxDM_COC')
        make_trials(ctxdm_olc, ctxdm_info, ctxdm_ntrials, base_dir + '/ctxDM_OLC')
        make_trials(ctxdm_lol, ctxdm_info, ctxdm_ntrials, base_dir + '/ctxDM_LOL')
    elif args.task == 'InterDMS':
        make_trials(DMSCategory, interdms_info, interdms_ntrials, base_dir + '/InterDMS_CTG_ABCABC', task_type='ABCABC')
        make_trials(DMSObject, interdms_info, interdms_ntrials, base_dir + '/InterDMS_OBJ_ABCABC', task_type='ABCABC')
        make_trials(DMSLoc, interdms_info, interdms_ntrials, base_dir + '/InterDMS_LOC_ABBCCA', task_type='ABBCCA')
        make_trials(DMSCategory, interdms_info, interdms_ntrials, base_dir + '/InterDMS_CTG_ABBCCA', task_type='ABBCCA')
        make_trials(DMSObject, interdms_info, interdms_ntrials, base_dir + '/InterDMS_OBJ_ABBCCA', task_type='ABBCCA')
        make_trials(DMSLoc, interdms_info, interdms_ntrials, base_dir + '/InterDMS_LOC_ABCABC', task_type='ABCABC')
    elif args.task == 'OnebackAndOr': 
        make_trials(DMSA_LC, oneback_andor_info, oneback_ntrials, base_dir + '/OnebackA_LC', k=4)
        make_trials(DMSA_LO, oneback_andor_info, oneback_ntrials, base_dir + '/OnebackA_LO', k=4)
        make_trials(DMSO_LC, oneback_andor_info, oneback_ntrials, base_dir + '/OnebackO_LC', k=4)
        make_trials(DMSO_LO, oneback_andor_info, oneback_ntrials, base_dir + '/OnebackO_LO', k=4)
    elif args.task == 'Twoback':
        make_trials(DMSLoc, twoback_info, twoback_ntrials, base_dir + '/Twoback_LOC', k=3)
        make_trials(DMSCategory, twoback_info, twoback_ntrials, base_dir + '/Twoback_CTG', k=3)
        make_trials(DMSObject, twoback_info, twoback_ntrials, base_dir + '/Twoback_OBJ', k=3)

