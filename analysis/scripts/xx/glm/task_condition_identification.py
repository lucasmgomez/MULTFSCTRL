#!/usr/bin/env python3
# This script is for **trial-level GLM outputs** and groups betas into task-defined conditions.
"""
Task-condition identification & grouping for **trial-level** GLM outputs.

Data layout per run (your pipeline):
  - HDF5:  <...>/<sub>/<ses>/func/<base>_betas.h5   with dataset `betas` shape (P x K_trials)
  - CSV:   sibling <base>_design.csv describing each trial regressor column, including at least:
           [trialNumber, stim_order, location, object, category, is_correct, type]

What it does
------------
• Reads per-run **trial-level** betas and the matching design CSV.
• Defines within-trial pairs by task (e.g., ctxdm: (1,1),(2,2),(3,3); interdms: (1,2),(2,3),(3,4), …).
• Builds a condition key from the **encoding rows** (so keys reflect features during presentation).
• Stores, by default, the **delay beta that follows the second stimulus (i2)** of each pair.
  (This is configurable via --target_selection.)

• Aggregates across all runs into: {condition_key: [beta_vec, beta_vec, ...]}.
• Saves a pickle of that dict + a repetition-count histogram.

CLI knobs
---------
--target_selection:
    - "delay_of_second"  (DEFAULT): use the delay beta that follows the second stimulus (i2)
    - "encoding_first"             : use the encoding beta of the first stimulus (your previous behavior)
--pair_selection (only used when target_selection=encoding_first):
    - "first" (DEFAULT) : store encoding of s1
    - "mean"            : store mean(encoding s1, encoding s2)

Default paths
-------------
  --glm_root   /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas
  --out_root   /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas

Outputs
-------
  <out_root>/grouped_betas/task_relevant_only/<subj>_task_condition_betas.pkl
  <out_root>/results/repetition_count/task_relevant_only/all_consecutive_stim_pairs/<subj>_repetition_distribution.png

Example
-------
python task_condition_identification.py \
  --subj sub-01 \
  --glm_root /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas \
  --out_root /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas \
  --tasks nback_loc \
  --correct_only \
  --target_selection delay_of_second
"""
from __future__ import annotations

import h5py
import pickle
from pathlib import Path
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# Helpers
# -----------------------

import re
from typing import Optional

_CFNAME_RE = re.compile(
    r"^sub-(?P<subj>\d+)_ses-(?P<ses>\d{3})_task-(?P<task>[A-Za-z0-9]+)_run-(?P<run>\d{2})_events\.tsv$"
)

def _digits(s) -> Optional[int]:
    """Return int from a string or int-like; None if not parseable."""
    if s is None:
        return None
    if isinstance(s, int):
        return s
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else None

def _normalize_ids(subj_id, ses, run):
    subj = _digits(subj_id)
    ses_i = _digits(ses)
    run_i = _digits(run)
    # build zero-padded strings
    subj_str = f"{subj:02d}" if subj is not None else None
    ses_str  = f"{ses_i:03d}" if ses_i is not None else None
    run_str  = f"{run_i:02d}" if run_i is not None else None
    return subj, ses_i, run_i, subj_str, ses_str, run_str

def _strip_block_suffix(block_name: str) -> str:
    """Turn 'interdms_obj_ABBA_block_2' -> 'interdms_obj_ABBA'."""
    return re.sub(r"_block_\d+$", "", block_name)

def _parse_converted_name(cf: str):
    """Extract (subj, ses, task, run) ints/strs from converted_file_name."""
    m = _CFNAME_RE.match(cf)
    if not m:
        return None
    d = m.groupdict()
    return {
        "subj": int(d["subj"]),
        "ses": int(d["ses"]),
        "task": d["task"],
        "run": int(d["run"]),
    }
def _canonicalize_design_lookup_df(df: pd.DataFrame) -> pd.DataFrame:
    """Trim spaces from headers & normalize expected columns for the design *lookup* table."""
    df = df.copy()
    # strip header whitespace
    df.columns = df.columns.str.strip()

    # some files might have weird spacing inside values too
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()

    # enforce the three columns we expect (after stripping)
    expected = {"session", "block_file_name", "converted_file_name"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Design lookup table is missing expected columns: {missing}. "
                         f"Found columns: {list(df.columns)}")

    return df


def get_detailed_task_name(
    subj_id,
    ses,
    task_name: str,
    run,
    design_file_df: pd.DataFrame,
    *,
    strict: bool = True,
) -> str:
    """
    Return the detailed task name (e.g., 'ctxdm_col', 'interdms_obj_ABBA', 'nback_loc')
    for the given (subj_id, ses, task_name, run) by looking it up in design_file_df.

    Matching is done by parsing design_file_df['converted_file_name'] and comparing
    numeric session/run (with zero-padding handled) and task_name literally.

    If multiple matches are found, the first unique block_file_name is used unless strict=True,
    in which case a ValueError is raised.
    """
    design_file_df = _canonicalize_design_lookup_df(design_file_df)
    # Normalize inputs and also build the canonical converted filename
    subj, ses_i, run_i, subj_str, ses_str, run_str = _normalize_ids(subj_id, ses, run)
    if ses_i is None or run_i is None:
        raise ValueError("Could not parse numeric session/run from inputs.")
    task_name = str(task_name)

    # Parse every converted_file_name once (cached in a temporary view)

    parsed = design_file_df["converted_file_name"].map(_parse_converted_name)
    mask = parsed.notna()

    # Apply filters: session, task, run
    mask &= parsed.map(lambda d: d["ses"] == ses_i if d else False)
    mask &= parsed.map(lambda d: d["task"] == task_name if d else False)
    mask &= parsed.map(lambda d: d["run"] == run_i if d else False)

    candidates = design_file_df.loc[mask, ["block_file_name", "converted_file_name"]]

    if candidates.empty:
        # As a fallback, try exact string equality on a canonical name (handles zero padding)
        canonical = f"sub-{subj_str if subj_str else ''}_ses-{ses_str}_task-{task_name}_run-{run_str}_events.tsv"
        fallback = design_file_df.loc[
            design_file_df["converted_file_name"] == canonical,
            ["block_file_name", "converted_file_name"],
        ]
        if fallback.empty:
            raise LookupError(
                f"No row found for ses={ses_i}, task='{task_name}', run={run_i}."
            )
        candidates = fallback

    # Ensure uniqueness / resolve duplicates
    unique_blocks = candidates["block_file_name"].drop_duplicates()
    if len(unique_blocks) > 1 and strict:
        raise ValueError(
            "Multiple matching rows found with different block_file_name values:\n"
            + "\n".join(unique_blocks.tolist())
        )

    block_name = unique_blocks.iloc[0]
    return _strip_block_suffix(block_name)


def parse_run_tokens(beta_file: Path):
    """Extract (subj, ses, task_name, run, base_stem) from a betas filename.
    Expects pattern like: sub-01_ses-003_task-ctxdm_run-01_betas.h5
    """
    stem = beta_file.stem  # remove .h5
    base = stem[:-6] if stem.endswith("_betas") else stem
    parts = base.split('_')
    subj = next((p for p in parts if p.startswith('sub-')), 'sub-unknown')
    ses  = next((p for p in parts if p.startswith('ses-')), 'ses-unknown')
    task = next((p for p in parts if p.startswith('task-')), 'task-unknown')
    run  = next((p for p in parts if p.startswith('run-')), 'run-unknown')
    task_name = task.split('-', 1)[1] if '-' in task else task
    return subj, ses, task_name, run, base


def determine_task_type(task_name: str):
    """Return list of (stim_order_i, stim_order_j) pairs per trial for a task."""
    t = task_name.lower()
    if t.startswith("interdms"):
        return [(1, 2), (2, 3), (3, 4)]
        # return [(1, 2)]
    elif t.startswith("ctxdm"):
        # self-pairs (1,1), (2,2), (3,3) as in your current setup
        return [(1, 2), (2, 3)]
        # return [(1, 2)]
    elif t.startswith("nback") or t.startswith("1back"):
        # consecutive items in a 6-item sequence (adjust if your sequence length differs)
        return [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        # return [(1, 2)]
    else:
        # sensible default
        return [(1, 2)]


def determine_feature_type(task_name: str):
    """Which features define a condition key for the task. Subset of {'location','object'}."""
    tn = task_name.lower()
    if ("obj" in tn) or ("ctg" in tn):
        return ["object"]
    elif ("loc" in tn) or ("location" in tn) or ("ctxdm" in tn):
        return ["location", "object"]
    # fallback
    return ["object"]


def build_condition_key(task_name: str, s1_row: pd.Series, s2_row: pd.Series, feature_type: list[str]) -> str:
    if ("object" in feature_type) and ("location" in feature_type):
        return (f"{task_name}_loc{s1_row['location']}_obj{s1_row['object']}"
                f"*loc{s2_row['location']}_obj{s2_row['object']}")
    elif ("object" in feature_type):
        return f"{task_name}_obj_{s1_row['object']}*obj_{s2_row['object']}"
    else:
        return f"{task_name}_trialpair"


def _canonicalize_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map possible aliases to canonical names used below."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    cols_lower = {c.lower(): c for c in df.columns}

    def first_present(*names):
        for n in names:
            if n in cols_lower:
                return cols_lower[n]
        return None

    rename_map = {}

    # stim_order
    src = first_present('stim_order', 'stimorder', 'stim-order')
    if src and src != 'stim_order':
        rename_map[src] = 'stim_order'

    # location
    src = first_present('location', 'loc', 'locmod', 'location_label')
    if src and src != 'location':
        rename_map[src] = 'location'

    # object
    src = first_present('object', 'obj', 'objmod', 'object_name', 'objectid')
    if src and src != 'object':
        rename_map[src] = 'object'

    # category
    src = first_present('category', 'ctg', 'ctgmod')
    if src and src != 'category':
        rename_map[src] = 'category'

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def group_betas_within_trial(
    beta_file: Path,
    cond_csv: Path,
    task_name: str,
    correct_only: bool,
    target_selection: str,
    pair_selection: str,
):
    """Group betas into condition buckets per trial.

    target_selection:
        - "delay_of_second": use delay that follows the second stimulus (i2)
        - "encoding_first":  use encoding of first stimulus (s1); pair_selection controls 'first' vs 'mean'
    """
    # Load betas (P x K_trials)
    # print(f"reading betas from {beta_file}")
    with h5py.File(beta_file, 'r') as f:
        if 'betas' not in f:
            raise ValueError(f"No 'betas' in {beta_file}")
        betas = f['betas'][()]
    print(f"Loaded betas shape: {betas.shape}")


    # Load design (one row per trial regressor IN ORDER)
    df = pd.read_csv(cond_csv)
    print(f"loaded design with {len(df)} rows from {cond_csv}")
    df = _canonicalize_design_columns(df)
    # Basic sanity
    required_cols = {"trialNumber", "stim_order", "type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Design CSV missing columns {missing} in {cond_csv}")

    # Optional: only correct trials if present
    print(f"require_correct={correct_only}")
    if correct_only and 'is_correct' in df.columns:
        def is_negative(v):
            if isinstance(v, (bool, np.bool_)):
                return (v is False) or (v == False)
            if pd.isna(v):
                return False
            sv = str(v).strip().lower()
            return sv in {"false", "b", "0"}
        df = df[~df['is_correct'].apply(is_negative)].copy()

        print(f"after filtering to correct trials, {len(df)} rows remain")
    # Ensure column index matches betas' columns
    df = df.reset_index(drop=True)
    df['col_index'] = np.arange(len(df), dtype=int)

    # stim_order as numeric for matching
    df['stim_order'] = pd.to_numeric(df['stim_order'], errors='coerce')
    df = df.dropna(subset=['stim_order'])
    df['stim_order'] = df['stim_order'].astype(int)


    if betas.shape[1] != len(df):
        warnings.warn(
            f"Betas columns ({betas.shape[1]}) != design rows ({len(df)}). Proceeding but mapping may be off."
        )

    pair_indices_list = determine_task_type(task_name)
    feature_type = determine_feature_type(task_name)
    print(f"Determined {len(pair_indices_list)} pair types for task {task_name}: {pair_indices_list}")
    print(f"Using feature_type={feature_type} for condition keys")

    condition_dict: dict[str, list[np.ndarray]] = defaultdict(list)

    print(df.head())
    # Group within trial
    g = df.groupby('trialNumber')
    for _, trial_df in g:
        print(f"processing trialNumber={trial_df['trialNumber'].iloc[0]} with {len(trial_df)} rows")
        # robust phase/type matching
        types_lower = trial_df['type'].astype(str).str.lower()

        enc_df   = trial_df[types_lower.str.contains('encod', na=False)]

        delay_df = trial_df[types_lower.str.contains('delay',  na=False)]
        print(f"len of enc_df={len(enc_df)}, delay_df={len(delay_df)}")

        for i1, i2 in pair_indices_list:
            # Encoding rows to build the key (features from presentation time)
            r1_enc = enc_df[enc_df['stim_order'] == i1]
            r2_enc = enc_df[enc_df['stim_order'] == i2]

            if r1_enc.empty or r2_enc.empty:
                print(f"  skipping pair ({i1},{i2}) due to missing encoding rows")
                continue

            s1 = r1_enc.iloc[0]
            s2 = r2_enc.iloc[0]
           
            key = build_condition_key(task_name, s1, s2, feature_type)
            print(f"  identified condition key: {key}")
            if target_selection == "delay_of_second":
                # Delay row that follows the *second* stimulus (i2)
                r2_del = delay_df[delay_df['stim_order'] == i2]
                
                assert len(r2_del) <= 1
                if r2_del.empty:
                    # No matching delay row — skip this pair
                    print(f"  skipping pair ({i1},{i2}) due to missing delay row")
                    continue
                s2d = r2_del.iloc[0]
                b = betas[:, int(s2d['col_index'])]
            else:
                # encoding_first (previous behavior)
                if pair_selection == 'mean':
                    b = 0.5 * (
                        betas[:, int(s1['col_index'])] +
                        betas[:, int(s2['col_index'])]
                    )
                else:  # 'first'
                    b = betas[:, int(s1['col_index'])]

            condition_dict[key].append(b)


    return condition_dict # condition_key -> list of beta vectors => need to check where i accumulate repetitions


# -----------------------
# Main
# -----------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Group trial-level betas into task-defined conditions")
    ap.add_argument("--subj", required=True, help="Subject ID, e.g., sub-01")
    ap.add_argument("--glm_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas",
                    help="Root of trial-level GLM outputs (contains <subj>/ses-*/func/*_betas.h5)")
    ap.add_argument("--out_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas",
                    help="Root for outputs (grouped pickle + histogram)")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Restrict to these tasks (e.g., ctxdm interdms nback)")
    ap.add_argument("--correct_only", action="store_true",
                    help="Use only correct trials if column present")

    # NEW: selection knobs
    ap.add_argument("--target_selection",
                    choices=["delay_of_second", "encoding_first"],
                    default="delay_of_second",
                    help="What beta to store for each (i1,i2) pair. Default: delay_of_second.")
    ap.add_argument("--pair_selection",
                    choices=["first", "mean"],
                    default="first",
                    help="Only used when --target_selection=encoding_first. 'first' uses s1, 'mean' averages s1 & s2.")

    args = ap.parse_args()

    subj = args.subj
    # read design_file_df
    design_file_df = pd.read_csv(f"/project/def-pbellec/xuan/fmri_dataset_project/data/study_designs/{subj}_design_design_with_converted.tsv", sep="\t")
    glm_root = Path(args.glm_root) / subj

    if not glm_root.exists():
        raise SystemExit(f"GLM root not found: {glm_root}")

    all_condition_groups: dict[str, list[np.ndarray]] = defaultdict(list)

    for session_dir in sorted(glm_root.glob("ses-*/")):
        func_dir = session_dir / "func"
        if not func_dir.exists():
            print(f"{func_dir} does not exist; skipping session")
            continue

        for beta_file in sorted(func_dir.glob("*_betas.h5")):
            # print(f"Reading {beta_file}")
            subj_id, ses, task_name, run, base = parse_run_tokens(beta_file)
            detailed_task_name = get_detailed_task_name(subj_id, ses, task_name, run, design_file_df, strict=True)
            task_name = detailed_task_name  # use the more specific name
            # print(f"task_name resolved to: {task_name}")
            # print(f"set(args.tasks)={set(args.tasks)}")
            # print(f"what is args.tasks={args.tasks}")
            # print(f"task_name={task_name}")
            # print(task_name not in set(args.tasks))
            if args.tasks and (task_name not in set(args.tasks)):
                continue

            base_clean = base.replace("_nilearn", "")

            # prefer design without nilearn suffix; try a few fallbacks
            cond_csv = beta_file.with_name(f"{base_clean}_design.csv")
            # print(f"processing task: {task_name}, run: {run}")
            if not cond_csv.exists():
                alt_csvs = [
                    beta_file.with_name(f"{base_clean}_nilearn_design.csv"),
                    beta_file.with_name(f"{base}_design.csv"),
                    beta_file.with_name(f"{base}_nilearn_design.csv"),
                ]
                cond_csv = next((p for p in alt_csvs if p.exists()), None)
                if cond_csv is None:
                    print(f"⚠️ Missing design CSV for {beta_file.name}; skipping run")
                    continue

            try:
                print("processing for task:", task_name)
                condition_betas = group_betas_within_trial(
                    beta_file=beta_file,
                    cond_csv=cond_csv,
                    task_name=task_name,
                    correct_only=args.correct_only,
                    target_selection=args.target_selection,
                    pair_selection=args.pair_selection,
                )
                print(f"successfully processed: {task_name}")
                for k, v in condition_betas.items():
                    all_condition_groups[k].extend(v)
            except Exception as e:
                print(f"⚠️ Skipping {beta_file.name}: {e}")

    # Report
    print("\n✅ Condition counts:")
    counts = {k: len(v) for k, v in all_condition_groups.items()}
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{k}: {v} repetitions")

    total_repetitions = sum(counts.values())
    print(f"\n🔢 Total conditions: {len(counts)}")
    print(f"🔁 Total repetitions across all conditions: {total_repetitions}")
    print(f"⚠️ Conditions with <5 repetitions: {sum(1 for v in counts.values() if v < 5)}")

    # Save grouped betas
    save_dir = Path(args.out_root) / "grouped_betas" / "task_relevant_only"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{subj}_task_condition_betas.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_condition_groups, f)
    print(f"\n✅ Grouped betas saved to: {save_path}")

    # Plot repetition distribution
    output_dir = (Path(args.out_root) / "results" / "repetition_count" /
                  "task_relevant_only" / "all_consecutive_stim_pairs")
    output_dir.mkdir(parents=True, exist_ok=True)

    repetition_counts = list(counts.values()) or [0]
    plt.figure(figsize=(10, 6))
    bins = range(1, (max(repetition_counts) if repetition_counts else 1) + 2)
    plt.hist(repetition_counts, bins=bins, edgecolor='black')
    plt.title(f"{subj} | Distribution of Repetitions per Condition")
    plt.xlabel("Number of Repetitions")
    plt.ylabel("Number of Conditions")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = output_dir / f"{subj}_repetition_distribution.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"📊 Repetition distribution figure saved to: {fig_path}")


if __name__ == "__main__":
    main()
