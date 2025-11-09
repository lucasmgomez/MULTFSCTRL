#!/usr/bin/env python3
import re
import h5py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.linear_model import LinearRegression
import nibabel as nib
from nilearn.glm.first_level import spm_hrf

from utils import glm_confounds_construction, standardize_run_label
import glob

RUN_RE = re.compile(r"run-(\d+)")

def run_number(run_str: str) -> int | None:
    m = RUN_RE.search(run_str)
    return int(m.group(1)) if m else None

def run_variants(run_str: str) -> list[str]:
    n = run_number(run_str)
    if n is None:
        return [run_str]
    return [f"run-{n:02d}", f"run-{n}"]  # try zero-padded and non-padded

def resolve_dtseries_path(fmri_root_subj: Path, ses: str, subj: str, task: str, run_str: str) -> Path | None:
    # Try both run-XX and run-X
    for rv in run_variants(run_str):
        cand = fmri_root_subj / ses / f"{subj}_{ses}_task-{task}_{rv}_space-Glasser64k_bold.dtseries.nii"
        if cand.exists():
            return cand
    # Fallback: glob any run-*, pick the one with same integer
    n = run_number(run_str)
    patt = str(fmri_root_subj / ses / f"{subj}_{ses}_task-{task}_run-*_space-Glasser64k_bold.dtseries.nii")
    for path in sorted(glob.glob(patt)):
        m = RUN_RE.search(path)
        if m and int(m.group(1)) == n:
            return Path(path)
    return None

def resolve_confounds_path(conf_root_subj: Path, ses: str, subj: str, task: str, run_str: str) -> Path | None:
    # fMRIPrep typically uses non-padded "run-1", but resolve robustly
    for rv in run_variants(run_str):
        cand = conf_root_subj / ses / "func" / f"{subj}_{ses}_task-{task}_{rv}_desc-confounds_timeseries.tsv"
        if cand.exists():
            return cand
    # Fallback: glob by run-*, match by integer
    n = run_number(run_str)
    patt = str(conf_root_subj / ses / "func" / f"{subj}_{ses}_task-{task}_run-*_desc-confounds_timeseries.tsv")
    for path in sorted(glob.glob(patt)):
        m = RUN_RE.search(path)
        if m and int(m.group(1)) == n:
            return Path(path)
    return None


# -----------------------
# CLI
# -----------------------
def get_args():
    p = argparse.ArgumentParser(description="Concatenate runs per task_type (from study_designs) and run GLM.")
    p.add_argument("--subj", required=True, help="e.g., sub-01")
    p.add_argument("--tr", type=float, default=1.49)
    p.add_argument("--tmask", type=int, default=1, help="Frames to drop at the start of each run")
    p.add_argument("--correct_only", action="store_true")

    # Roots
    p.add_argument("--fmri_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled")
    p.add_argument("--conf_root", default="/project/def-pbellec/xuan/cneuromod.multfs.fmriprep")
    p.add_argument("--events_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior")
    p.add_argument("--designs_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/study_designs")
    p.add_argument("--out_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data")

    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# -----------------------
# Helpers
# -----------------------
def default_output_root(out_root_base: Path, correct_only: bool, subj: str) -> Path:
    return (out_root_base / ("correct_betas" if correct_only else "betas") / subj)

def read_tasktype_map(designs_root: Path, subj: str) -> dict:
    """
    Read .../study_designs/<subj>_design_design_with_converted.tsv
    Return mapping: converted_file_name -> task_type (prefix of block_file_name before '_block_').
    """
    tsv = designs_root / f"{subj}_design_design_with_converted.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"Study design TSV not found: {tsv}")

    df = pd.read_csv(tsv, sep=r"\s*\t\s*", engine="python")
    # Normalize whitespace
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].str.strip()

    if "block_file_name" not in df.columns or "converted_file_name" not in df.columns:
        raise ValueError("Expected columns 'block_file_name' and 'converted_file_name' in study design TSV.")

    mapping = {}
    for _, row in df.iterrows():
        block = str(row["block_file_name"]).strip()
        conv = str(row["converted_file_name"]).strip()
        # task_type = substring before "_block_"
        m = re.split(r"_block_\d+", block)
        task_type = m[0] if m and m[0] else block
        mapping[conv] = task_type
    return mapping  # { "sub-01_ses-001_task-ctxdm_run-01_events.tsv": "ctxdm_col", ... }


def list_events_grouped_by_tasktype(events_root_subj: Path, mapping: dict) -> dict:
    """
    Scan events files under <events_root>/<subj>/ses-*/func/*.tsv, group by task_type using mapping.
    Returns: dict[task_type] -> list of dicts with keys: ses, run, task_name, events_path
    """
    groups = {}
    # enumerate all events that appear in mapping (safer than scanning everything)
    for conv_name, task_type in mapping.items():
        # expected events path is .../<subj>/<ses>/func/<conv_name>
        # We don't know ses upfront, conv_name itself has ses-XXX, so find by glob:
        # But faster is to split conv_name to extract ses/run/task
        m = re.match(r"(sub-\d+)_([^_]+)_task-([^_]+)_(run-[^_]+)_events\.tsv", conv_name)
        if not m:
            # fallback: glob
            matches = list(events_root_subj.rglob(conv_name))
            if not matches:
                continue
            events_path = matches[0]
            # try to parse
            mm = re.match(rf"{re.escape(events_root_subj.name)}/(ses-[^/]+)/func/(.+)", str(events_path.relative_to(events_root_subj)).replace("\\","/"))
            # If parsing failed, just skip
            if not mm:
                continue
            ses = mm.group(1)
            # Extract pieces from filename
            fn = events_path.name
            m2 = re.match(rf"{re.escape(events_root_subj.name)}_(ses-[^_]+)_task-([^_]+)_(run-[^_]+)_events\.tsv", f"{events_root_subj.name}_{ses}_{fn}")
            if not m2:
                continue
            _, ses, task_name, run = m2.groups()
        else:
            subj, ses, task_name, run = m.groups()
            events_path = events_root_subj / ses / "func" / conv_name

        if not events_path.exists():
            # Not all mapping lines correspond to actual files on disk
            continue

        groups.setdefault(task_type, []).append({
            "ses": ses, "run": run, "task_name": task_name, "events_path": events_path
        })
    # Sort runs deterministically
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda d: (d["ses"], d["run"]))
    return groups


def load_confounds(conf_file: Path) -> np.ndarray:
    df_conf = pd.read_csv(conf_file, sep="\t")
    C = glm_confounds_construction(df_conf)
    return np.nan_to_num(C, nan=0.0).astype(np.float32)


def clean_events(df_events: pd.DataFrame) -> pd.DataFrame:
    df = df_events.copy()
    df.columns = df.columns.str.strip()
    # Remove spaces in string cells (applymap deprecated)
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
    return df


def build_trial_regressors(df_events: pd.DataFrame, num_trs: int, tr_length: float) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Per-trial regressors (T x Ntrials), convolved with canonical SPM HRF.
    Returns (regressors_hrf, design_per_trial)
    """
    # basic checks
    required = {"onset_time", "offset_time"}
    if not required.issubset(df_events.columns):
        raise ValueError(f"Events missing required columns: {required - set(df_events.columns)}")

    regs = []
    meta = { "regressor_type": [], "stim_order": [], "trialNumber": [] }
    for _, row in df_events.iterrows():
        reg = np.zeros((num_trs,), dtype=np.float32)
        onset = int(np.ceil(row["onset_time"] / tr_length))
        offset = int(np.floor(row["offset_time"] / tr_length))
        onset = np.clip(onset, 0, num_trs)
        offset = np.clip(offset, 0, num_trs)
        if offset > onset:
            reg[onset:offset] = 1.0
        regs.append(reg)
        # carry some columns if they exist
        meta["regressor_type"].append(row.get("type", row.get("regressor_type", "NA")))
        meta["stim_order"].append(row.get("stim_order", np.nan))
        meta["trialNumber"].append(row.get("trialNumber", np.nan))

    R = np.stack(regs, axis=1) if regs else np.zeros((num_trs, 0), dtype=np.float32)

    # Convolve
    hrf = spm_hrf(tr_length, oversampling=1)
    R_hrf = np.zeros_like(R, dtype=np.float32)
    for j in range(R.shape[1]):
        tmp = np.convolve(R[:, j], hrf)
        R_hrf[:, j] = tmp[:num_trs]

    design_df = pd.DataFrame(meta)
    return R_hrf, design_df


def block_diag_concat(R_list: list[np.ndarray]) -> np.ndarray:
    """
    Construct block-diagonal concatenation of per-run trial regressor matrices.
    Each R_i is (T_i x K_i). Output is (sum T_i) x (sum K_i).
    """
    T_total = sum(R.shape[0] for R in R_list)
    K_total = sum(R.shape[1] for R in R_list)
    out = np.zeros((T_total, K_total), dtype=np.float32)
    t0 = 0
    k0 = 0
    for R in R_list:
        T_i, K_i = R.shape
        out[t0:t0+T_i, k0:k0+K_i] = R
        t0 += T_i
        k0 += K_i
    return out


def run_glm(timeseries: np.ndarray, confounds: np.ndarray, regressors_hrf: np.ndarray):
    """
    Fit OLS: Y = Xb, where X = [confounds, regressors_hrf]
    Return (betas_task_only, all_regressors, residuals_transposed)
    """
    X = np.hstack([confounds, regressors_hrf]).astype(np.float32)
    Y = timeseries.astype(np.float32)
    reg = LinearRegression().fit(X, Y)
    betas_full = reg.coef_                 # (P x (C+K))
    K = regressors_hrf.shape[1]
    betas_task = betas_full[:, -K:]
    y_pred = reg.predict(X)
    resid = Y - y_pred                     # (T x P)
    return betas_task, X, resid.T


# -----------------------
# Main per-task_type concat
# -----------------------
def main():
    args = get_args()
    subj = args.subj

    fmri_root_subj = Path(args.fmri_root) / subj
    conf_root_subj = Path(args.conf_root) / subj
    events_root_subj = Path(args.events_root) / subj
    out_root_subj = default_output_root(Path(args.out_root), args.correct_only, subj)

    # 1) Read study design and get mapping converted_file_name -> task_type
    mapping = read_tasktype_map(Path(args.designs_root), subj)

    # 2) Group actual on-disk events by task_type
    groups = list_events_grouped_by_tasktype(events_root_subj, mapping)
    task_types = sorted(groups.keys())
    if not task_types:
        print(f"[⚠️] No task_types found for {subj}. Is the TSV correct and events present?")
        return

    print(f"[ℹ️] Subject {subj}: found task_types: {task_types}")

    # 3) For each task_type, collect runs across sessions, build run-wise pieces, then concatenate and GLM
    for task_type in task_types:
        runs = groups[task_type]
        if not runs:
            continue

        print(f"[▶] Task type: {task_type} | {len(runs)} run(s)")

        TS_list = []
        C_list = []
        R_list = []
        D_list = []   # design rows per trial, we’ll add run/session info too

        for item in runs:
            ses = item["ses"]
            run = item["run"]
            base_task = item["task_name"]  # e.g., 'ctxdm', 'interdms'
            events_path = item["events_path"]

            # OLD (problematic)
            # ts_file = fmri_root_subj / ses / f"{subj}_{ses}_task-{base_task}_{run}_space-Glasser64k_bold.dtseries.nii"
            # conf_file = conf_root_subj / ses / "func" / f"{subj}_{ses}_task-{base_task}_{run}_desc-confounds_timeseries.tsv"

            # NEW (robust)
            ts_file = resolve_dtseries_path(fmri_root_subj, ses, subj, base_task, run)
            conf_file = resolve_confounds_path(conf_root_subj, ses, subj, base_task, run)

            if ts_file is None:
                print(f"[⚠️] Missing timeseries (tried run variants) for {subj} {ses} {base_task} {run}, skipping this run.")
                continue
            if conf_file is None:
                print(f"[⚠️] Missing confounds (tried run variants) for {subj} {ses} {base_task} {run}, skipping this run.")
                continue

            if not ts_file.exists():
                print(f"[⚠️] Missing timeseries: {ts_file}, skipping this run.")
                continue
            if not conf_file.exists():
                print(f"[⚠️] Missing confounds: {conf_file}, skipping this run.")
                continue

            # Load data
            timeseries = nib.load(str(ts_file)).get_fdata(dtype=np.float32)
            df_conf = load_confounds(conf_file)
            df_events = pd.read_csv(events_path, sep="\t")
            df_events = clean_events(df_events)

            # Filter correctness if requested
            if args.correct_only and "is_correct" in df_events.columns:
                df_events = df_events[df_events["is_correct"] == True]

            num_trs = timeseries.shape[0]

            # Per-run trial design
            R_hrf, design_trial = build_trial_regressors(df_events, num_trs, args.tr)

            # tmask skip
            keep = np.ones((num_trs,), dtype=bool)
            keep[:args.tmask] = False
            timeseries = timeseries[keep, :]
            df_conf = df_conf[keep, :]
            R_hrf = R_hrf[keep, :]

            # demean/detrend
            timeseries = signal.detrend(timeseries, axis=0, type="constant")
            timeseries = signal.detrend(timeseries, axis=0, type="linear")

            # Append
            TS_list.append(timeseries)
            C_list.append(df_conf)
            R_list.append(R_hrf)

            # annotate design with ses/run/task_type for traceability
            design_trial = design_trial.copy()
            design_trial["ses"] = ses
            design_trial["run"] = run
            design_trial["task_base"] = base_task
            design_trial["task_type"] = task_type
            D_list.append(design_trial)

        if not TS_list:
            print(f"[⚠️] No usable runs for task_type {task_type}, skipping.")
            continue

        # Concatenate across runs
        TS_cat = np.vstack(TS_list)                 # (sum_T x P)
        C_cat = np.vstack(C_list)                   # (sum_T x C)
        R_cat = block_diag_concat(R_list)           # (sum_T x sum_K)
        D_cat = pd.concat(D_list, ignore_index=True)

        # Fit GLM
        betas_task, X_full, resid_TP = run_glm(TS_cat, C_cat, R_cat)  # betas_task: (P x sum_K)

        # Save under .../betas/<subj>/task-<task_type>/concat/
        target_dir = out_root_subj / f"task-{task_type}" / "concat"
        target_dir.mkdir(parents=True, exist_ok=True)
        base = f"{subj}_task-{task_type}_ses-ALL_run-ALL"

        # Save design CSV (trial-level, concatenated, with ses/run)
        design_csv = target_dir / f"{base}_design.csv"
        D_cat.to_csv(design_csv, index=False)

        # Save H5: betas (task only) + all_regressors (confounds+task)
        h5_file = target_dir / f"{base}_betas.h5"
        if h5_file.exists() and not args.overwrite:
            print(f"[⏩] Exists, skipping (use --overwrite to replace): {h5_file}")
        else:
            with h5py.File(h5_file, "w") as h5f:
                h5f.create_dataset("betas", data=betas_task.astype(np.float32))         # (P x K_total)
                h5f.create_dataset("all_regressors", data=X_full.astype(np.float32))    # (sum_T x (C + K_total))
                # helpful metadata
                h5f.attrs["task_type"] = np.string_(task_type)
                h5f.attrs["regressor_level"] = np.string_("trial")
                # this tells you total per-trial columns in the same order as rows of D_cat
                # (we don't store names because trials are usually unnamed; D_cat rows are the mapping)

        print(f"[🎯] Saved {h5_file} and {design_csv}")

    print("[✅] Done.")


if __name__ == "__main__":
    main()
