import sys
sys.path.append('/home/lucas/projects/iWISDM')

from iwisdm import make
from iwisdm import read_write
import json
import os
import random
from pathlib import Path
from copy import deepcopy
from numpy.random import choice 


stim_dir = '/mnt/tempdata/lucas/multi_task_data/stim/only_fmri_zoom' 
output_dir = '/home/lucas/projects/MULTFSCTRL/analysis/scripts/task_search/trials'

# Create environment
env = make(
    env_id='ShapeNet',
    dataset_fp=stim_dir,
)
# Initialize environment
print(env.env_spec.MAX_DELAY)

# Get all task filepaths
task_dir = '/home/lucas/projects/MULTFSCTRL/analysis/scripts/task_search/base_tasks'
all_tasks = [os.path.join(task_dir, f) for f in os.listdir(task_dir) if f.endswith('.json')]

# Parameters for merging
max_len = 7
stop_factor = 0.1  # Probability of stopping merging at each step

def generate_merged_trial(idx:int, merged_task = None):
    
    # Randomly merge tasks
    if merged_task is None:
        merged_info, merged_fps = random_merge()
    else:
        merged_info, merged_fps = merged_task

    
    # Generate trial
    try:
        imgs, _, info_dict = merged_info.generate_trial(canvas_size=224,
                                                    fixation_cue=True,
                                                    cue_on_action=True,
                                                    stim_data=env.stim_data.splits['train']['data'],
                                                    return_objset=False,)

        # Create trial directory
        fp = os.path.join(output_dir, 'trial' + str(idx))
        if not os.path.exists(fp):
            os.makedirs(fp)

        # Write trial
        read_write.write_trial(imgs, info_dict, fp)

        # Check if successful, if not try again
        if 'task_info.json' not in os.listdir(fp + '/frames'):
            print('Error: task_info.json not found in ', fp)
            generate_merged_trial(idx)
        elif len(imgs) != len(info_dict['answers']):
            print('Error: Number of images and answers do not match in ', fp) # NOTE: This is a temporary fix for a merge bug which causes multiple actions assigned to one frame
            generate_merged_trial(idx)
        else:
            # Load task_info.json
            task_info = json.load(open(fp + '/frames/task_info.json'))

            # Add merged to task_info
            task_info['merged'] = True

            # Add task paths to task_info
            task_info['task_paths'] = [path.split('/')[-1] for path in merged_fps]

            # Write task_info.json
            with open(fp + '/frames/task_info.json', 'w') as f:
                json.dump(task_info, f)

            # Create a done and ready flag file
            Path(os.path.join(fp, 'frames', 'DONE.bin')).touch()

    except:
        print('Error: Could not generate merged trial with tasks ', merged_fps)
        generate_merged_trial(idx, (merged_info, merged_fps))
    return


def random_merge():
    taskA_fp = random.choice(all_tasks)
    taskA = env.read_task(taskA_fp)

    taskA_info = env.init_compositional_tasks([taskA])[0]
    taskA_n_frames = taskA_info.n_epochs

    # Randomly set first_shareable for taskA
    taskA_info.frame_info.first_shareable = random.randint(0, taskA_n_frames)

    frames_left = max_len - taskA_n_frames

    # Randomly pick a task from all_tasks
    counter = 0
    merged_fps = [taskA_fp]
    while frames_left > 0:
        # Random stopping mechanism
        if random.random() < stop_factor:
            print('Triggered early stop')
            break
        
        taskB_fp = random.choice(all_tasks)
        taskB = env.read_task(taskB_fp)

        # Init taskB and get number of frames
        taskB_info = env.init_compositional_tasks([taskB])[0]
        taskB_n_frames = taskB_info.n_epochs

        # Randomly set first_shareable for taskB based on frames_left
        taskB_info.frame_info.first_shareable = random.randint(0, taskB_n_frames-1)
        
        # Check if taskB can be merged with taskA
        taskA_info_copy = deepcopy(taskA_info)
        taskB_info_copy = deepcopy(taskB_info)
        taskA_info_copy.merge(taskB_info_copy)
        if taskA_info_copy.n_epochs > max_len:
            continue

        # Merge tasks
        taskA_info.merge(taskB_info)

        import pdb; pdb.set_trace()
        if taskA_info.frame_info.objset.end_epoch != list(range(0, taskA_info.n_epochs+1, 2)):
            continue

        # Update frames_left
        frames_left -= taskA_info.n_epochs

        counter += 1

        # Add taskB_fp to merged_fps
        merged_fps.append(taskB_fp)

    return taskA_info, merged_fps

"""
Random trial generation
"""

n_trials = 250

for i in range(n_trials):
    print('Generating merged trial ', i)
    generate_merged_trial(i)    
