import re
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


def count_interdms_task_conditions(task_name: str, df: pd.DataFrame) -> dict:
    """
    Count the number of repetitions for each task condition based on task name and trial structure.

    Parameters:
        task_name (str): e.g. 'interdms_obj_ABAB', 'interdms_loc_ABBA'
        df (pd.DataFrame): must include columns 'trialNumber', 'stim_order', 'location', 'object', 'category'

    Returns:
        dict: condition -> number of repetitions
    """
    # only consider encoding trials
    df = df[df["regressor_type"] == "encoding"] 

    # Determine task rule and pair indices
    if task_name.endswith("ABAB"):
        pair_indices_list = [(1, 3), (2, 4)]
    elif task_name.endswith("ABBA"):
        pair_indices_list = [(1, 4), (2, 3)]
    else:
        raise ValueError("Task name must end with either 'ABAB' or 'ABBA'")

    # Determine task feature type
    if "obj" in task_name:
        feature_type = "object"
    elif "ctg" in task_name:
        feature_type = "category"
    elif "loc" in task_name:
        feature_type = "location_object"  # special handling
    else:
        raise ValueError("Task name must contain one of 'obj', 'ctg', or 'loc'")

    grouped = df.groupby("trialNumber")
    condition_counter = Counter()

    for trial_num, trial_df in grouped:
        for pair_indices in pair_indices_list:
            stim1 = trial_df[trial_df["stim_order"] == pair_indices[0]]
            stim2 = trial_df[trial_df["stim_order"] == pair_indices[1]]
            if stim1.empty or stim2.empty:
                continue  # skip incomplete pairs

            s1 = stim1.iloc[0]
            s2 = stim2.iloc[0]

            if feature_type == "object":
                condition = f"obj_{s1['object']}*obj_{s2['object']}"
            elif feature_type == "category":
                condition = f"obj_{s1['object']}*obj_{s2['object']}"
            elif feature_type == "location_object":
                condition = f"loc{s1['location']}_obj{s1['object']}*loc{s2['location']}_obj{s2['object']}"
            else:
                raise RuntimeError("Unrecognized feature_type")

            condition_counter[condition] += 1

    return dict(condition_counter)

def merge_and_add(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged:
            merged[key] += value
        else:
            merged[key] = value
    return merged


all_tasks = ["interdms_obj_ABAB", "interdms_obj_ABBA", "interdms_loc_ABAB", "interdms_loc_ABBA", "interdms_ctg_ABAB", "interdms_ctg_ABBA"]
# Set subject ID and paths
subject = "sub-01"
betas_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/betas")
metadata_path = Path(f"/project/def-pbellec/xuan/fmri_dataset_project/data/study_designs/{subject}_design_design_with_converted.tsv")


for target_task in all_tasks:
    print(f"Processing task: {target_task}")
    results = {}
    
    # Load and clean the metadata
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    metadata_df.columns = metadata_df.columns.str.strip()
    metadata_df = metadata_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Locate subject-specific beta path
    betas_path = betas_dir / subject

    # Find all CSV files that contain "interdms" in the name
    design_csvs = list(betas_path.rglob("*interdms*_design.csv"))

    # Iterate over matched CSVs
    for csv_path in design_csvs:
        filename = csv_path.name
        match = re.match(rf"{subject}_(ses-\d+)_task-interdms_run-(\d+)_design.csv", filename)
        if not match:
            print(f"Filename format not recognized: {filename}")
            continue

        ses_str, run_str = match.groups()
        session_num = int(ses_str.split('-')[1])
        run_num = int(run_str)

        # Construct the corresponding events.tsv filename
        events_name = f"{subject}_{ses_str}_task-interdms_run-{run_num:02d}_events.tsv"

        # Find matching row in metadata
        row = metadata_df[
            (metadata_df["session"] == session_num) &
            (metadata_df["converted_file_name"] == events_name)
        ]

        if row.empty:
            print(f"No metadata match found for {events_name}")
            continue

        block_file = row.iloc[0]["block_file_name"]
        # Remove '_block' and the trailing number
        task_name = re.sub(r'_block_\d+$', '', block_file)

        print(f"filename: {filename}, task_name: {task_name}")

        if task_name == target_task:
            # Construct full path to events file and read it
            events_path = Path(f"/project/def-pbellec/xuan/fmri_dataset_project/data/betas/{subject}/{ses_str}/func/{events_name[:-10]}design.csv")
            try:
                df = pd.read_csv(events_path)
                df.columns = df.columns.str.strip()
                df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            except FileNotFoundError:
                print(f"Events file not found: {events_path}")
                raise FileNotFoundError(f"Events file not found: {events_path}")
            

            # Count repetitions
            condition_counts = count_interdms_task_conditions(task_name, df)
            results = merge_and_add(results, condition_counts)


    for condition, count in results.items():
        print(f"{condition}: {count}")



    # Create output directory if it doesn't exist
    output_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/results/repetition_count/all_stim_pairs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count how many conditions have fewer than 5 repetitions
    total_conditions = len(results)
    low_count_conditions = sum(1 for count in results.values() if count < 4)
    low_ratio = low_count_conditions / total_conditions if total_conditions > 0 else 0

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(results.keys(), results.values())
    plt.xticks(rotation=90)
    plt.ylabel("Repetition Count")
    plt.xlabel("Task Condition")

    title = f"{target_task}_trial_repetition_count (below 4: {low_count_conditions}/{total_conditions}, ratio={low_ratio:.2f})"
    plt.title(title)

    # Save figure
    fig_path = output_dir / f"{subject}_{target_task}_trial_repetition_count.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    print(f"Plot saved to {fig_path}")
