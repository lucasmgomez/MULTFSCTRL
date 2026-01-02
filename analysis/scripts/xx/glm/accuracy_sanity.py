# sanity check: interdms ABAB task, the action targets are wrong [was generated in the format of ABBA]
# reformat the target actions to match the expected ABAB format



# step 1: make a copy of all behavioral data
# import os
# import shutil
# from pathlib import Path
# import glob

# # Define source and destination root paths
# source_root = Path("/project/def-pbellec/xuan/cneuromod.multfs.fmriprep/sourcedata/cneuromod.multfs.raw")
# destination_root = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/behavior")

# # Define subjects and sessions
# subjects = ['01', '02', '03', '05', '06']
# sessions = [f"{i:03}" for i in range(1, 17)]  # 001 to 016

# for sub in subjects:
#     for ses in sessions:
#         # Build path to the func directory
#         func_dir = source_root / f"sub-{sub}" / f"ses-{ses}" / "func"
#         if not func_dir.exists():
#             raise FileNotFoundError(f"Function directory does not exist: {func_dir}")
#             continue 
        
#         # Find all matching event TSV files
#         pattern = f"sub-{sub}_ses-{ses}_task-*_run-*_events.tsv"
#         matching_files = list(func_dir.glob(pattern))
        
#         for file_path in matching_files:
#             # Define corresponding destination path
#             dest_dir = destination_root / f"sub-{sub}" / f"ses-{ses}" / "func"
#             dest_dir.mkdir(parents=True, exist_ok=True)

#             # Copy the file
#             shutil.copy(file_path, dest_dir / file_path.name)
#             print(f"Copied: {file_path} -> {dest_dir / file_path.name}")


# step 2: udpate design file to find exact task type
# import pandas as pd
# from pathlib import Path
# import re

# # Directory containing the design files
# design_root = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/study_designs")
# output_suffix = "_design_with_converted.tsv"

# # Function to identify the task type from the block file name
# def identify_task_type(filename):
#     filename = filename.lower()
#     if "ctxdm" in filename:
#         return "ctxdm"
#     elif "interdms" in filename:
#         return "interdms"
#     elif "nback" in filename or "1back" in filename:
#         return "1back"
#     else:
#         raise ValueError(f"Unknown task type in file: {filename}")

# # Loop through all subject design files
# for design_file in sorted(design_root.glob("sub-*_design.tsv")):
#     subject = re.search(r"sub-(\d+)", design_file.name).group(0)  # e.g., sub-01
#     df = pd.read_csv(design_file, sep="\t")

#     # Prepare list to store new filenames
#     converted_file_names = []

#     # Process by session
#         # Process by session
#     for ses, ses_df in df.groupby("session"):
#         ses_padded = f"{int(ses):03d}"  # Ensure session ID is zero-padded
#         task_counters = {"ctxdm": 0, "interdms": 0, "1back": 0}

#         for i, row in ses_df.iterrows():
#             task = identify_task_type(row["block_file_name"])
#             task_counters[task] += 1
#             run_id = task_counters[task]
#             converted = f"{subject}_ses-{ses_padded}_task-{task}_run-{run_id:02d}_events.tsv"
#             converted_file_names.append(converted)

#     # Add to DataFrame and save
#     df["converted_file_name"] = converted_file_names
#     output_file = design_file.with_name(design_file.stem + output_suffix)
#     df.to_csv(output_file, sep="\t", index=False)
#     print(f"✅ Saved: {output_file}")


# step 3: update ABAB task actions
import pandas as pd
from pathlib import Path
import pdb

behavior_root = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/behavior")
design_root = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/study_designs")
# subjects = ['01', '02', '03', '05', '06']
subjects = ['05']


def update_actions_rowwise(df, task_type):
    """
    Update action1 and action2 columns row-by-row based on task type.
    """
    action1_list = []
    action2_list = []
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

    for _, row in df.iterrows():
        if task_type == "obj":
            action1 = row["objmod1"] == row["objmod3"]
            action2 = row["objmod2"] == row["objmod4"]
        elif task_type == "loc":
            action1 = row["locmod1"] == row["locmod3"]
            action2 = row["locmod2"] == row["locmod4"]
        elif task_type == "ctg":
            action1 = row["ctgmod1"] == row["ctgmod3"]
            action2 = row["ctgmod2"] == row["ctgmod4"]
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        action1_list.append(action1)
        action2_list.append(action2)

    df["action1"] = action1_list
    df["action2"] = action2_list
    return df


for sub in subjects:

    subj_str = f"sub-{sub}"

    # Load design file with converted_file_name
    design_file = design_root / f"{subj_str}_design_design_with_converted.tsv"
    if not design_file.exists():
        print(f"[Warning] Design file missing: {design_file}")
        continue
    design_df = pd.read_csv(design_file, sep="\t")
    design_df.columns = design_df.columns.str.strip()
    # Remove all spaces from string values in the DataFrame
    design_df = design_df.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

    # Look for interdms behavioral files
    for path in (behavior_root / subj_str).rglob(f"{subj_str}_ses-*_task-interdms_run-*_events.tsv"):
        try:
            file_name = path.name
          
           
            ses = path.parts[-3]  # ses-XXX

            # Extract session number padded to 3 digits
            ses_number = ses.replace("ses-", "")
            if ses_number.isdigit():
                ses_padded = f"{int(ses_number):03d}"
            else:
                print(f"[Warning] Cannot parse session: {ses}")
                continue
            
            
            # Match design row by session and converted file name
            row = design_df.query(f'converted_file_name == "{file_name}"')
            
            # row = design_df.query(f"session == '{int(ses_number)}' and converted_file_name == '{file_name}'")
            if row.empty:
                print(f"[Warning] No match in design for {file_name}")
                continue

            block_name = row.iloc[0]["block_file_name"].lower()
            
            # Determine task subtype
            if block_name.startswith("interdms_obj_abab"):
                task_type = "obj"
            elif block_name.startswith("interdms_loc_abab"):
                task_type = "loc"
            elif block_name.startswith("interdms_ctg_abab"):
                task_type = "ctg"
            else:
                print(f"[Warning] Unrecognized interdms subtype for {file_name}: {block_name}")
                continue

            # Load and update behavioral file row-by-row
            df = pd.read_csv(path, sep="\t")
            df = update_actions_rowwise(df, task_type)

            # Save updated file
            df.to_csv(path, sep="\t", index=False)
            print(f"✅ Updated: {file_name} [{task_type}]")

        except Exception as e:
            print(f"[❌ Error] Failed to process {path}: {e}")
