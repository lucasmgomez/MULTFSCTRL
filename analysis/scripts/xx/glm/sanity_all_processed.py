import os
from pathlib import Path
import pandas as pd
import warnings  

warnings.filterwarnings("ignore", category=UserWarning)

# Set subject and task
subject = "sub-03"
all_tasks = ["interdms_obj_ABAB", "interdms_obj_ABBA", "interdms_loc_ABAB", "interdms_loc_ABBA", "interdms_ctg_ABAB", "interdms_ctg_ABBA"]
for task_name in all_tasks:
    # Load design metadata
    design_file_path = f"/project/def-pbellec/xuan/fmri_dataset_project/data/study_designs/{subject}_design_design_with_converted.tsv"
    design_df = pd.read_csv(design_file_path, sep='\t')
    design_df.columns = design_df.columns.str.strip()
    design_df = design_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Filter rows where block_file_name starts with target task
    filtered_df = design_df[design_df["block_file_name"].str.startswith(task_name)]

    # Check each expected beta file
    for _, row in filtered_df.iterrows():
        converted_filename = row["converted_file_name"]  # e.g., sub-01_ses-001_task-interdms_run-01_events.tsv

        # Replace '_events.tsv' with '_betas.h5'
        beta_filename = converted_filename.replace("_events.tsv", "_betas.h5")

        # Extract session to build path
        parts = beta_filename.split("_")
        ses_part = [p for p in parts if p.startswith("ses-")][0]  # e.g., 'ses-001'
        betas_root = Path(f"/project/def-pbellec/xuan/fmri_dataset_project/data/betas/{subject}")
        session_dir = betas_root / ses_part / "func"
        beta_path = session_dir / beta_filename

        # Check existence
        if not beta_path.exists():
            print(f"❌🐛💥Missing beta file: {beta_path}")
    