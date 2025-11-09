import os
import pandas as pd
from collections import Counter

def find_ctxdm_conditions(base_dir):
    trial_conditions = []

    # Walk through directory to find ctxdm .tsv files
    for root, _, files in os.walk(base_dir):
        for file in files:
            if "ctxdm" in file and file.endswith(".tsv"):
                tsv_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(tsv_path, sep='\t')
                    if 'objlocmod1' in df.columns and 'objlocmod2' in df.columns:
                        for _, row in df.iterrows():
                            condition = (row['objlocmod1'], row['objlocmod2'])
                            trial_conditions.append(condition)
                    else:
                        print(f"Skipping {tsv_path}: missing expected columns.")
                except Exception as e:
                    print(f"Error reading {tsv_path}: {e}")

    return Counter(trial_conditions)

def main():
    base_dir = "/project/def-pbellec/xuan/cneuromod.multfs.fmriprep/sourcedata/cneuromod.multfs.raw/sub-03"
    condition_counts = find_ctxdm_conditions(base_dir)

    # Convert results to DataFrame
    df = pd.DataFrame(condition_counts.items(), columns=["Condition (objlocmod1, objlocmod2)", "Num Trials"])
    df.sort_values(by="Num Trials", ascending=False, inplace=True)

    print("\n=== Trial Condition Counts ===")
    print(df)

    print(f"\nTotal unique conditions: {len(condition_counts)}")
    print(f"Total trials: {sum(condition_counts.values())}")

if __name__ == "__main__":
    main()
