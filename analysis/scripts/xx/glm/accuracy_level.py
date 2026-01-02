# evaluate response accuracy based on the target action
import pandas as pd
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def evaluate_accuracy_1back(data):
    total_correct = 0
    total_trials = 0
    data.columns = data.columns.str.strip()
    data = data.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)


    for _, row in data.iterrows():
        for i in range(1, 6):
            action_col = f"action{i + 1}"
            response_col = f"response_{i}"
            if action_col in row and response_col in row:
                action = row[action_col]
                response = row[response_col]
                if action == "True": action = True
                elif action == "False": action = False
                if (action is True and response == 'x') or (action is False and response == 'b'):
                    total_correct += 1
                total_trials += 1

    return total_correct / total_trials if total_trials > 0 else float('nan')

def evaluate_accuracy_dms(data):
    """
    Evaluate the accuracy of responses in a DMS task.

    Parameters:
    data: DataFrame, should contain:
        - action: boolean (True = match, False = non-match)
        - response_1: string ('x' = match response, 'b' = non-match response)

    Returns:
    float: Proportion of correct responses across all rows.
    """
    total_correct = 0
    total_trials = 0
    data.columns = data.columns.str.strip()
    data = data.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)


    for _, row in data.iterrows():
        if 'action' in row and 'response_1' in row:
            action = row['action']
            response = row['response_1']
            if action == "True": action = True
            elif action == "False": action = False

            # Correct if match->b or non-match->x
            if (action is True and response == 'x') or (action is False and response == 'b'):
                total_correct += 1
            total_trials += 1

    return total_correct / total_trials if total_trials > 0 else float('nan')

def evaluate_accuracy_ctxdm(data):
    """
    Evaluate the accuracy of responses in a CTX-DM task.

    Parameters:
    data: DataFrame, should contain:
        - action: boolean (True = match, False = non-match)
        - response_2: string ('x' = match response, 'b' = non-match response)

    Returns:
    float: Proportion of correct responses across all rows.
    """
    total_correct = 0
    total_trials = 0
    data.columns = data.columns.str.strip()
    data = data.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)


    for _, row in data.iterrows():
        if 'action' in row and 'response_2' in row:
            action = row['action']
            response = row['response_2']
            if action == "True": action = True
            elif action == "False": action = False

            if (action is True and response == 'x') or (action is False and response == 'b'):
                total_correct += 1
            total_trials += 1

    return total_correct / total_trials if total_trials > 0 else float('nan')

def evaluate_accuracy_interdms(data):
    """
    Evaluate the accuracy of responses in an interleaved DMS (interdms) task.

    Parameters:
    data: DataFrame, should contain:
        - action1, action2: boolean (True = match, False = non-match)
        - response_2, response_3: string ('b' = match response, 'x' = non-match)

    Returns:
    float: Proportion of correct responses across all rows and both targets.
    """
    total_correct = 0
    total_trials = 0
    # Remove all spaces from string values in the DataFrame
    data.columns = data.columns.str.strip()
    data = data.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

    for _, row in data.iterrows():
        for i in range(1, 3):  # action1/2 and response_2/3
            action_col = f"action{i}"
            response_col = f"response_{i + 1}"
            

            if action_col in row and response_col in row:
                action = row[action_col]
                response = row[response_col]
                if action == "True": action = True
                elif action == "False": action = False
                # print(f"action type: {type(action)}")
                # print(f"action: {action}, response: {response}")  # Debugging output
                if (action is True and response == 'x') or (action is False and response == 'b'):
                    total_correct += 1
                total_trials += 1

    return total_correct / total_trials if total_trials > 0 else float('nan')


root_path = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/behavior")
subjects = ['01', '02', '03', '05', '06']

sessions = [f"{i:03}" for i in range(1, 17)]
task_name = "interdms"

# Store results
results = []

for sub in subjects:
    for ses in sessions:
        func_dir = root_path / f"sub-{sub}" / f"ses-{ses}" / "func"
        if not func_dir.exists():
            print(f"[Warning] Missing directory: {func_dir}")
            continue 
        
        pattern = f"sub-{sub}_ses-{ses}_task-{task_name}_run-*_events.tsv"
        matching_files = list(func_dir.glob(pattern))
        
        for file_path in matching_files:
            # print(file_path.type)
            # if str(file_path) != "/project/def-pbellec/xuan/fmri_dataset_project/data/behavior/sub-06/ses-016/func/sub-06_ses-016_task-interdms_run-02_events.tsv":
            #     continue
            # print(f"Processing file: {file_path}")
            data = pd.read_csv(file_path, sep='\t')
            if task_name == "1back":
                if len(data) != 9:  # optional sanity check
                    print(f"[Warning] Unexpected number of trials in {file_path}: {len(data)}")
                accuracy = evaluate_accuracy_1back(data)
            elif task_name == "dms":
                if len(data) != 16:  # optional sanity check
                    print(f"[Warning] Unexpected number of trials in {file_path}: {len(data)}")
                accuracy = evaluate_accuracy_dms(data)
            elif task_name == "ctxdm":
                if len(data) != 20:  # optional sanity check
                    print(f"[Warning] Unexpected number of trials in {file_path}: {len(data)}")
                accuracy = evaluate_accuracy_ctxdm(data)
            elif task_name == "interdms":
                if len(data) != 16:  # optional sanity check
                    print(f"[Warning] Unexpected number of trials in {file_path}: {len(data)}")
                accuracy = evaluate_accuracy_interdms(data)
            print(f"Accuracy for subject: {sub}, session: {ses}: {accuracy:.2f}")

            results.append({
                "subject": f"sub-{sub}",
                "session": f"ses-{ses}",
                "task": task_name,
                "run": file_path.name.split("_run-")[1].split("_")[0],  # extract run number
                "accuracy": accuracy
            })

# Save as CSV
results_df = pd.DataFrame(results)
results_csv_path = Path(f"/project/def-pbellec/xuan/fmri_dataset_project/results/{task_name}_accuracy_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\n✅ Accuracy results saved to: {results_csv_path}")
