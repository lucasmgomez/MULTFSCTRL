# do not consider dmsloc for now since it only contains locmod


# this function reformats the behavior data from the 1-back task into a specific format. it takes dataframe as input. 
# for each row within each run, I want to reformat the data into the following format:
# columns: trialNumber, stim_order, loc/ctg/objmod, type[encoding, delay], is_correct, onset_time, offset_time. 
# for each stimuli, there will be two rows, one for encoding and one for delay. 
# for 1-back task:
# there will be 6 stimuli per trial, that intotal gives us 12 rows.
# starting from first stimuli [stim_order: 1; locmod: locmod1, ctdmod: ctgmod1, objmod: objmod1]
# for the first row [type: "encoding"], is_correct will be marked as none. 
# onset_time will be stimulus_0_onset, offset_time will be stimulus_0_offset.
# for the second row [type: "delay"], is_correct will be marked as action2 == response_1
# onset_time will be stimulus_0_offset, offset_time will be stimulus_0_offset + 2TR. (TR = 1.49sec)
# etc until the last stimulus.
# this function returns a dataframe with the above format.

import pandas as pd
from pathlib import Path
MAP = {"x": True,
        "b": False,
        None: None,
        "--": None,}

def reformat_ctxdm_behavior(df, tr=1.49):
    """
    Reformat ctxdm task behavioral data into long-format with encoding and delay phases.

    Parameters:
    - df: pandas DataFrame. One row per trial.
    - tr: float. Repetition time for computing delay offset time.

    Returns:
    - new_df: pandas DataFrame with columns:
      ['trialNumber', 'stim_order', 'locmod', 'ctgmod', 'objmod', 'type',
       'is_correct', 'onset_time', 'offset_time']
    """
    output_rows = []
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

    for _, row in df.iterrows():
        trial_number = row.get("TrialNumber", None)
        assert trial_number is not None, "TrialNumber column is required in the input DataFrame"

        for i in range(3):  # ctxdm has 3 stimuli
            stim_order = i + 1
            locmod = row[f'locmod{stim_order}']
            ctgmod = row[f'ctgmod{stim_order}']
            objmod = row[f'objmod{stim_order}']
            onset = row[f'stimulus_{i}_onset']
            offset = row[f'stimulus_{i}_offset']

            # Encoding row
            output_rows.append({
                'trialNumber': trial_number,
                'stim_order': stim_order,
                'locmod': locmod,
                'ctgmod': ctgmod,
                'objmod': objmod,
                'type': 'encoding',
                'is_correct': None,
                'onset_time': onset,
                'offset_time': offset
            })

            # Delay row
            if stim_order < 3:
                is_correct = None
            else:
                is_correct = row.get('action') == MAP[row.get('response_2', None)]

            output_rows.append({
                'trialNumber': trial_number,
                'stim_order': stim_order,
                'locmod': locmod,
                'ctgmod': ctgmod,
                'objmod': objmod,
                'type': 'delay',
                'is_correct': is_correct,
                'onset_time': offset,
                'offset_time': offset + 2 * tr
            })

    new_df = pd.DataFrame(output_rows)
    return new_df


def reformat_interdms_behavior(df, tr=1.49):
    """
    Reformat interDMS task behavioral data into long-format with encoding and delay phases.

    Parameters:
    - df: pandas DataFrame. One row per trial.
    - tr: float. Repetition time for computing delay offset time.

    Returns:
    - new_df: pandas DataFrame with columns:
      ['trialNumber', 'stim_order', 'locmod', 'ctgmod', 'objmod', 'type',
       'is_correct', 'onset_time', 'offset_time']
    """
    output_rows = []
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

    for _, row in df.iterrows():
        trial_number = row.get("TrialNumber", None)
        assert trial_number is not None, "TrialNumber column is required in the input DataFrame"

        for i in range(4):  # 4 stimuli
            stim_order = i + 1
            locmod = row[f'locmod{stim_order}']
            ctgmod = row[f'ctgmod{stim_order}']
            objmod = row[f'objmod{stim_order}']
            onset = row[f'stimulus_{i}_onset']
            offset = row[f'stimulus_{i}_offset']

            # Encoding row
            output_rows.append({
                'trialNumber': trial_number,
                'stim_order': stim_order,
                'locmod': locmod,
                'ctgmod': ctgmod,
                'objmod': objmod,
                'type': 'encoding',
                'is_correct': None,
                'onset_time': onset,
                'offset_time': offset
            })

            # Delay row with correct logic
            if stim_order < 3:
                is_correct = None
            else:
                response_index = stim_order - 1  # response_2 for stim 3, response_3 for stim 4
                is_correct = row.get(f'action{stim_order-2}') == MAP[row.get(f'response_{response_index}', None)]

            output_rows.append({
                'trialNumber': trial_number,
                'stim_order': stim_order,
                'locmod': locmod,
                'ctgmod': ctgmod,
                'objmod': objmod,
                'type': 'delay',
                'is_correct': is_correct,
                'onset_time': offset,
                'offset_time': offset + 2 * tr
            })

    new_df = pd.DataFrame(output_rows)
    return new_df


def reformat_1back_behavior(df, tr=1.49):
    """
    Reformat 1-back task behavioral data into long-format with encoding and delay phases.

    Parameters:
    - df: pandas DataFrame. One row per trial.
    - tr: float. Repetition time for computing delay offset time.

    Returns:
    - new_df: pandas DataFrame with columns:
      ['trialNumber', 'stim_order', 'locmod', 'ctgmod', 'objmod', 'type',
       'is_correct', 'onset_time', 'offset_time']
    """
    output_rows = []
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

    for _, row in df.iterrows():
        trial_number = row.get("TrialNumber", None)  # if trial number is a column
        assert trial_number is not None, "trialNumber column is required in the input DataFrame"
        
        for i in range(6):
            stim_order = i + 1
            locmod = row[f'locmod{i+1}']
            ctgmod = row[f'ctgmod{i+1}']
            objmod = row[f'objmod{i+1}']
            onset = row[f'stimulus_{i}_onset']
            offset = row[f'stimulus_{i}_offset']

            # Encoding row
            output_rows.append({
                'trialNumber': trial_number,
                'stim_order': stim_order,
                'locmod': locmod,
                'ctgmod': ctgmod,
                'objmod': objmod,
                'type': 'encoding',
                'is_correct': None,
                'onset_time': onset,
                'offset_time': offset
            })

            # Delay row
            if i == 0:
                is_correct = None  # no previous response to compare to
            else:
                # print(f"row response: {row.get(f'response_{i}')}, action: {row[f'action{i+1}']}")
                is_correct = row[f'action{i+1}'] == MAP[row.get(f'response_{i}', None)]
                # print(f"inspection of delay frame {i}: action{i+1}={row[f'action{i+1}']}, response_{i}={MAP[row.get(f'response_{i}', None)]}, is_correct={is_correct}")
                assert is_correct is not None, f"response_{i} column is required for correctness check"
            
            output_rows.append({
                'trialNumber': trial_number,
                'stim_order': stim_order,
                'locmod': locmod,
                'ctgmod': ctgmod,
                'objmod': objmod,
                'type': 'delay',
                'is_correct': is_correct,
                'onset_time': offset,
                'offset_time': offset + 2 * tr
            })

    new_df = pd.DataFrame(output_rows)
    return new_df


root_path = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/behavior")
target_path = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior")
subjects = ['01', '02', '03', '05', '06']


sessions = [f"{i:03}" for i in range(1, 17)]

task_name = "ctxdm"  # change to "interdms" or "ctxdm" as needed
for sub in subjects:
    for ses in sessions:
        func_dir = root_path / f"sub-{sub}" / f"ses-{ses}" / "func"
        if not func_dir.exists():
            print(f"[Warning] Missing directory: {func_dir}")
            continue 
        
        pattern = f"sub-{sub}_ses-{ses}_task-{task_name}_run-*_events.tsv"
        matching_files = list(func_dir.glob(pattern))
        
        for file_path in matching_files:
            print(f"Processing file: {file_path}")
            data = pd.read_csv(file_path, sep='\t')
            if task_name == "1back":
                if len(data) != 9:  # optional sanity check
                    print(f"[Warning] Unexpected number of trials in {file_path}: {len(data)}")
                else:
                    reformatted_data = reformat_1back_behavior(data)
                    
                    # Preserve sub/ses/func hierarchy in target path
                    relative_path = file_path.relative_to(root_path)
                    target_file_path = target_path / relative_path
                    
                    # Create the target directory if needed
                    target_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save reformatted data
                    reformatted_data.to_csv(target_file_path, sep='\t', index=False)
                    print(f"Reformatted data saved to: {target_file_path}")
            elif task_name == "interdms":
                if len(data) != 16:  # optional sanity check
                    print(f"[Warning] Unexpected number of trials in {file_path}: {len(data)}")
                else:   
                    reformatted_data = reformat_interdms_behavior(data)
                    
                    # Preserve sub/ses/func hierarchy in target path
                    relative_path = file_path.relative_to(root_path)
                    target_file_path = target_path / relative_path
                    
                    # Create the target directory if needed
                    target_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save reformatted data
                    reformatted_data.to_csv(target_file_path, sep='\t', index=False)
                    print(f"Reformatted data saved to: {target_file_path}")
            elif task_name == "ctxdm":
                if len(data) != 20:  # optional sanity check
                    print(f"[Warning] Unexpected number of trials in {file_path}: {len(data)}")    
                else:
                    reformatted_data = reformat_ctxdm_behavior(data)
                    
                    # Preserve sub/ses/func hierarchy in target path
                    relative_path = file_path.relative_to(root_path)
                    target_file_path = target_path / relative_path
                    
                    # Create the target directory if needed
                    target_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save reformatted data
                    reformatted_data.to_csv(target_file_path, sep='\t', index=False)
                    print(f"Reformatted data saved to: {target_file_path}")